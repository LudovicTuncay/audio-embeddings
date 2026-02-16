import glob
import math
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import lightning as L
import torch
from torch.utils.data import DataLoader

from src.data.audio_utils import collate_audio_batch
from src.utils import RankedLogger

try:
    import webdataset as wds
except ImportError:
    wds = None

try:
    from torchcodec.decoders import AudioDecoder
except Exception as exc:
    AudioDecoder = None
    _TORCHCODEC_IMPORT_ERROR = exc
else:
    _TORCHCODEC_IMPORT_ERROR = None


log = RankedLogger(__name__, rank_zero_only=True)


def _raise_torchcodec_unavailable() -> None:
    raise RuntimeError(
        "UPSWebDatasetDataModule requires torchcodec AudioDecoder for decoding. "
        "Install torchcodec and ensure FFmpeg runtime libraries are available. "
        "See: https://github.com/pytorch/torchcodec#installing-torchcodec"
    ) from _TORCHCODEC_IMPORT_ERROR


def _coerce_audio_bytes(payload: Any) -> Optional[bytes]:
    """Converts WebDataset payload objects to bytes when possible."""
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, memoryview):
        return payload.tobytes()
    return None


class UPSSampleDecoder:
    """Decode one random fixed-length chunk from a raw WebDataset sample."""

    def __init__(self, target_sample_rate: int, chunk_sec: float):
        if AudioDecoder is None:
            _raise_torchcodec_unavailable()
        self.target_sample_rate = target_sample_rate
        self.chunk_sec = chunk_sec
        self.chunk_samples = int(target_sample_rate * chunk_sec)

    def _extract_audio_bytes(self, sample: Dict[str, Any]) -> Optional[bytes]:
        # UPS shards store audio as .mp3 files; WebDataset exposes that as `sample["mp3"]`.
        payload = sample.get("mp3")
        return _coerce_audio_bytes(payload)

    def _decode_with_torchcodec(
        self, audio_bytes: bytes
    ) -> tuple[torch.Tensor, float, float]:
        decoder = AudioDecoder(
            source=audio_bytes,
            sample_rate=self.target_sample_rate,
            num_channels=1,
        )

        metadata = getattr(decoder, "metadata", None)
        duration = getattr(metadata, "duration_seconds_from_header", None)
        duration = float(duration) if duration is not None else None

        if duration is None or not math.isfinite(duration) or duration <= 0.0:
            chunk_start_sec = 0.0
            chunk_duration_sec = self.chunk_sec
            chunk_end_sec = chunk_start_sec + chunk_duration_sec
        elif duration <= self.chunk_sec:
            chunk_start_sec = 0.0
            chunk_duration_sec = duration
            chunk_end_sec = duration
        else:
            max_start_sec = duration - self.chunk_sec
            chunk_start_sec = random.uniform(0.0, max_start_sec)
            chunk_duration_sec = self.chunk_sec
            chunk_end_sec = chunk_start_sec + self.chunk_sec

        decoded = decoder.get_samples_played_in_range(chunk_start_sec, chunk_end_sec)
        chunk = decoded.data

        if chunk.ndim == 2:
            chunk = chunk.squeeze(0)
        elif chunk.ndim > 2:
            chunk = chunk.reshape(-1)

        return chunk.to(dtype=torch.float32), chunk_start_sec, chunk_duration_sec

    def __call__(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        audio_bytes = self._extract_audio_bytes(sample)
        if audio_bytes is None:
            return None

        audio_name = str(sample.get("__key__", ""))
        source_url = str(sample.get("__url__", ""))

        try:
            chunk, chunk_start_sec, chunk_duration_sec = self._decode_with_torchcodec(
                audio_bytes
            )

            if chunk.shape[-1] < self.chunk_samples:
                pad_amount = self.chunk_samples - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_amount))
            elif chunk.shape[-1] > self.chunk_samples:
                chunk = chunk[: self.chunk_samples]

            waveform = chunk.unsqueeze(0)

            return {
                "waveform": waveform,
                "audio_name": audio_name,
                "source_url": source_url,
                "chunk_start_sec": float(chunk_start_sec),
                "chunk_duration_sec": float(chunk_duration_sec),
            }
        except Exception:
            return None


class UPSWebDatasetDataModule(L.LightningDataModule):
    """LightningDataModule for local UPS WebDataset shards."""

    def __init__(
        self,
        shard_globs: list[str],
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        target_sample_rate: int = 16000,
        chunk_sec: float = 10.0,
        collate_mode: str = "pad",
        val_num_shards: int = 32,
        test_num_shards: int = 32,
        split_seed: int = 42,
        train_shardshuffle: int = 200,
        train_sampleshuffle: int = 1000,
        eval_sampleshuffle: int = 0,
        persistent_workers: bool = True,
        drop_last: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        if AudioDecoder is None:
            _raise_torchcodec_unavailable()

        self.shard_globs = shard_globs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target_sample_rate = target_sample_rate
        self.chunk_sec = chunk_sec
        self.collate_mode = collate_mode
        self.val_num_shards = val_num_shards
        self.test_num_shards = test_num_shards
        self.split_seed = split_seed
        self.train_shardshuffle = train_shardshuffle
        self.train_sampleshuffle = train_sampleshuffle
        self.eval_sampleshuffle = eval_sampleshuffle
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last

        self.all_shards: list[str] = []
        self.train_shards: list[str] = []
        self.val_shards: list[str] = []
        self.test_shards: list[str] = []

        self.train_dataset: Optional[Iterable[Any]] = None
        self.val_dataset: Optional[Iterable[Any]] = None
        self.test_dataset: Optional[Iterable[Any]] = None

        self._splits_initialized = False
        self._split_logged = False

    @staticmethod
    def _shard_sort_key(path: str) -> tuple[int, int, str]:
        stem = Path(path).stem
        try:
            return (0, int(stem), path)
        except ValueError:
            return (1, 0, path)

    def _discover_shards(self) -> list[str]:
        discovered: list[str] = []
        for pattern in self.shard_globs:
            discovered.extend(glob.glob(pattern, recursive=True))

        shard_paths: list[str] = []
        for path_str in discovered:
            shard_path = Path(path_str)
            if not shard_path.is_file():
                continue
            if shard_path.suffix.lower() != ".tar":
                continue
            shard_paths.append(str(shard_path.resolve()))

        unique_shards = sorted(set(shard_paths), key=self._shard_sort_key)
        return unique_shards

    def _initialize_splits(self) -> None:
        self.all_shards = self._discover_shards()
        total_shards = len(self.all_shards)

        if total_shards == 0:
            raise ValueError(
                "No UPS shard files were found. Check `data.shard_globs` values."
            )

        holdout_shards = self.val_num_shards + self.test_num_shards
        if holdout_shards >= total_shards:
            raise ValueError(
                "Invalid holdout split: val_num_shards + test_num_shards must be "
                f"< total shards. Got val={self.val_num_shards}, "
                f"test={self.test_num_shards}, total={total_shards}."
            )

        shuffled = list(self.all_shards)
        rng = random.Random(self.split_seed)
        rng.shuffle(shuffled)

        test_end = self.test_num_shards
        val_end = self.test_num_shards + self.val_num_shards

        self.test_shards = shuffled[:test_end]
        self.val_shards = shuffled[test_end:val_end]
        self.train_shards = shuffled[val_end:]

        self._splits_initialized = True

    def _build_webdataset(
        self,
        shards: list[str],
        is_train: bool,
    ) -> Optional[Iterable[Any]]:
        if len(shards) == 0:
            return None

        if wds is None:
            raise ImportError(
                "webdataset is required for UPS streaming. Install with `uv sync`."
            )

        sample_decoder = UPSSampleDecoder(
            target_sample_rate=self.target_sample_rate,
            chunk_sec=self.chunk_sec,
        )

        dataset: Iterable[Any] = wds.WebDataset(
            shards,
            shardshuffle=self.train_shardshuffle if is_train else False,
            nodesplitter=wds.split_by_node,
            handler=wds.handlers.ignore_and_continue,
        )

        if is_train and self.train_sampleshuffle > 0:
            dataset = dataset.shuffle(self.train_sampleshuffle)
        elif not is_train and self.eval_sampleshuffle > 0:
            dataset = dataset.shuffle(self.eval_sampleshuffle)

        dataset = dataset.map(sample_decoder).select(lambda sample: sample is not None)
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        if not self._splits_initialized:
            self._initialize_splits()

        if not self._split_logged:
            log.info(
                "UPS shard split ready: "
                f"total={len(self.all_shards)} "
                f"train={len(self.train_shards)} "
                f"val={len(self.val_shards)} "
                f"test={len(self.test_shards)}"
            )
            self._split_logged = True

        if stage == "fit" or stage is None:
            self.train_dataset = self._build_webdataset(
                shards=self.train_shards, is_train=True
            )
            self.val_dataset = self._build_webdataset(
                shards=self.val_shards, is_train=False
            )

        if stage == "test" or stage is None:
            self.test_dataset = self._build_webdataset(
                shards=self.test_shards, is_train=False
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError(
                "Train dataset is not initialized. Call setup('fit') first."
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0 and self.persistent_workers,
            drop_last=self.drop_last,
            collate_fn=partial(self.collate_fn, mode=self.collate_mode),
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError(
                "Val dataset is not initialized. Call setup('fit') first."
            )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0 and self.persistent_workers,
            drop_last=False,
            collate_fn=partial(self.collate_fn, mode=self.collate_mode),
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError(
                "Test dataset is not initialized. Call setup('test') first."
            )

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0 and self.persistent_workers,
            drop_last=False,
            collate_fn=partial(self.collate_fn, mode=self.collate_mode),
        )

    @staticmethod
    def collate_fn(batch: list[Dict[str, Any]], mode: str = "pad") -> Dict[str, Any]:
        return collate_audio_batch(
            batch=batch,
            waveform_key="waveform",
            mode=mode,
        )
