import glob
import math
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.audio_utils import DatasetResamplerCropper, collate_audio_batch
from src.utils import RankedLogger

try:
    from datasets import Audio, load_dataset
except ImportError as exc:
    Audio = None
    load_dataset = None
    _DATASETS_IMPORT_ERROR = exc
else:
    _DATASETS_IMPORT_ERROR = None


log = RankedLogger(__name__, rank_zero_only=True)


class PeoplesSpeechDataset(Dataset):
    """Map-style wrapper over a HuggingFace split with AudioSet-compatible outputs."""

    def __init__(
        self,
        hf_split: Any,
        split_name: str,
        audio_column: str = "audio",
        id_column: Optional[str] = "id",
        text_column: Optional[str] = "text",
        max_length: Optional[int] = None,
        target_sample_rate: int = 16000,
        decode_error_policy: str = "skip",
    ):
        if decode_error_policy not in {"skip", "raise", "zero"}:
            raise ValueError(
                "decode_error_policy must be one of {'skip', 'raise', 'zero'}, "
                f"got '{decode_error_policy}'"
            )

        self.hf_split = hf_split
        self.split_name = split_name
        self.audio_column = audio_column
        self.id_column = id_column
        self.text_column = text_column
        self.max_length = max_length
        self.target_sample_rate = target_sample_rate
        self.decode_error_policy = decode_error_policy

        self.resampler = DatasetResamplerCropper(
            target_sr=target_sample_rate,
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.hf_split)

    @staticmethod
    def _to_mono_waveform(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            return waveform

        if waveform.ndim == 2:
            # Handle either [C, T] or [T, C] layouts.
            if waveform.shape[0] <= 8 and waveform.shape[0] <= waveform.shape[1]:
                return waveform.mean(dim=0)
            if waveform.shape[1] <= 8 and waveform.shape[1] < waveform.shape[0]:
                return waveform.mean(dim=1)
            return waveform.reshape(-1)

        return waveform.reshape(-1)

    def _audio_name(self, row: Dict[str, Any], idx: int, audio_obj: Any) -> str:
        if self.id_column and self.id_column in row and row[self.id_column] is not None:
            return str(row[self.id_column])

        if isinstance(audio_obj, dict):
            audio_path = audio_obj.get("path")
            if audio_path:
                return str(audio_path)

        return f"{self.split_name}-{idx}"

    def _decode_waveform(self, row: Dict[str, Any]) -> tuple[torch.Tensor, int, Any]:
        if self.audio_column not in row:
            raise KeyError(
                f"Missing audio column '{self.audio_column}'. "
                f"Available keys: {list(row.keys())}"
            )

        audio_obj = row[self.audio_column]
        if not isinstance(audio_obj, dict):
            raise TypeError(
                f"Expected dict in '{self.audio_column}', got {type(audio_obj)}"
            )

        waveform_array = audio_obj.get("array")
        source_sr = int(audio_obj.get("sampling_rate", self.target_sample_rate))

        if waveform_array is None:
            raise ValueError("Decoded audio array is None")

        waveform = torch.as_tensor(waveform_array, dtype=torch.float32)
        if waveform.numel() == 0:
            raise ValueError("Decoded audio array is empty")

        waveform = self._to_mono_waveform(waveform)
        return waveform, source_sr, audio_obj

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.hf_split[idx]

        try:
            waveform, source_sr, audio_obj = self._decode_waveform(row)
            waveform = self.resampler(waveform, source_sr=source_sr)

            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

            sample: Dict[str, Any] = {
                "waveform": waveform.to(dtype=torch.float32),
                "audio_name": self._audio_name(row=row, idx=idx, audio_obj=audio_obj),
                "index": idx,
            }

            if self.text_column and self.text_column in row:
                sample["text"] = row[self.text_column]

            return sample
        except Exception as exc:
            if self.decode_error_policy == "raise":
                raise RuntimeError(
                    f"Failed to decode sample idx={idx} from split '{self.split_name}'"
                ) from exc

            if self.decode_error_policy == "zero":
                fallback_length = (
                    self.max_length
                    if self.max_length is not None
                    else self.target_sample_rate
                )
                return {
                    "waveform": torch.zeros(1, fallback_length, dtype=torch.float32),
                    "audio_name": f"{self.split_name}-{idx}",
                    "index": idx,
                }

            return {
                "audio_name": f"{self.split_name}-{idx}",
                "index": idx,
                "error": True,
            }


class PeoplesSpeechDataModule(L.LightningDataModule):
    """LightningDataModule for PeoplesSpeech parquet splits backed by HF datasets."""

    def __init__(
        self,
        data_root: str,
        subset: str = "clean",
        cache_dir: Optional[str] = None,
        split_file_patterns: Optional[Dict[str, Sequence[str]]] = None,
        audio_column: str = "audio",
        id_column: Optional[str] = "id",
        text_column: Optional[str] = "text",
        duration_column_candidates: Optional[Sequence[str]] = None,
        min_duration_sec: Optional[float] = 10.0,
        max_duration_sec: Optional[float] = 30.0,
        max_audio_length_sec: Optional[float] = 10.0,
        target_sample_rate: int = 16000,
        decode_error_policy: str = "skip",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        collate_mode: str = "pad",
    ):
        super().__init__()
        self.save_hyperparameters()

        if decode_error_policy not in {"skip", "raise", "zero"}:
            raise ValueError(
                "decode_error_policy must be one of {'skip', 'raise', 'zero'}, "
                f"got '{decode_error_policy}'"
            )

        if prefetch_factor <= 0:
            raise ValueError(f"prefetch_factor must be > 0, got {prefetch_factor}")

        if min_duration_sec is not None and min_duration_sec < 0:
            raise ValueError(f"min_duration_sec must be >= 0, got {min_duration_sec}")

        if max_duration_sec is not None and max_duration_sec < 0:
            raise ValueError(f"max_duration_sec must be >= 0, got {max_duration_sec}")

        if (
            min_duration_sec is not None
            and max_duration_sec is not None
            and min_duration_sec > max_duration_sec
        ):
            raise ValueError(
                "min_duration_sec must be <= max_duration_sec; "
                f"got min={min_duration_sec}, max={max_duration_sec}"
            )

        self.data_root = os.path.expandvars(os.path.expanduser(data_root))
        self.subset = subset
        self.cache_dir = (
            os.path.expandvars(os.path.expanduser(cache_dir)) if cache_dir else None
        )

        self.split_file_patterns = split_file_patterns or {
            "train": ["train*.parquet"],
            "validation": ["valid*.parquet", "validation*.parquet"],
            "test": ["test*.parquet"],
        }

        self.audio_column = audio_column
        self.id_column = id_column
        self.text_column = text_column
        self.duration_column_candidates = list(
            duration_column_candidates
            if duration_column_candidates is not None
            else ["duration_ms", "duration_sec", "duration"]
        )

        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.target_sample_rate = target_sample_rate
        self.decode_error_policy = decode_error_policy

        if max_audio_length_sec is not None:
            self.max_audio_length = int(max_audio_length_sec * target_sample_rate)
        else:
            self.max_audio_length = None

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.collate_mode = collate_mode

        self.train_dataset: Optional[PeoplesSpeechDataset] = None
        self.val_dataset: Optional[PeoplesSpeechDataset] = None
        self.test_dataset: Optional[PeoplesSpeechDataset] = None

        self._wrapped_splits: Optional[Dict[str, PeoplesSpeechDataset]] = None

    def _subset_dir(self) -> Path:
        return Path(self.data_root) / self.subset

    def _discover_split_files(self, split_name: str) -> List[str]:
        patterns = self.split_file_patterns.get(split_name)
        if not patterns:
            raise ValueError(f"No patterns configured for split '{split_name}'")
        if isinstance(patterns, str):
            patterns = [patterns]

        subset_dir = self._subset_dir()
        matches: List[str] = []
        for pattern in patterns:
            expanded = os.path.expandvars(os.path.expanduser(pattern))
            if os.path.isabs(expanded):
                glob_pattern = expanded
            else:
                glob_pattern = str(subset_dir / expanded)

            matches.extend(glob.glob(glob_pattern))

        files = sorted({str(Path(path).resolve()) for path in matches})
        if not files:
            raise FileNotFoundError(
                f"No parquet files found for split '{split_name}' under '{subset_dir}'. "
                f"Patterns: {list(patterns)}"
            )

        return files

    def _build_data_files(self) -> Dict[str, List[str]]:
        return {
            "train": self._discover_split_files("train"),
            "validation": self._discover_split_files("validation"),
            "test": self._discover_split_files("test"),
        }

    @staticmethod
    def _to_positive_float(value: Any) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(number) or number <= 0.0:
            return None

        return number

    def _duration_to_seconds(self, value: Any, column_name: str) -> Optional[float]:
        duration = self._to_positive_float(value)
        if duration is None:
            return None

        if column_name.endswith("_ms"):
            return duration / 1000.0

        return duration

    def _resolve_duration_column(self, split_dataset: Any) -> Optional[str]:
        column_names = set(getattr(split_dataset, "column_names", []))
        for candidate in self.duration_column_candidates:
            if candidate in column_names:
                return candidate
        return None

    def _apply_duration_filter(self, split_dataset: Any, split_name: str) -> Any:
        if self.min_duration_sec is None and self.max_duration_sec is None:
            return split_dataset

        duration_col = self._resolve_duration_column(split_dataset)
        if duration_col is None:
            log.warning(
                "No duration column found for split '%s'. Candidates: %s. "
                "Skipping duration filter.",
                split_name,
                self.duration_column_candidates,
            )
            return split_dataset

        before_count = len(split_dataset)

        def keep_example(example: Dict[str, Any]) -> bool:
            duration_sec = self._duration_to_seconds(
                example.get(duration_col), duration_col
            )
            if duration_sec is None:
                return False

            if (
                self.min_duration_sec is not None
                and duration_sec < self.min_duration_sec
            ):
                return False
            if (
                self.max_duration_sec is not None
                and duration_sec > self.max_duration_sec
            ):
                return False
            return True

        split_dataset = split_dataset.filter(
            keep_example,
            desc=f"Filtering {split_name} by duration",
        )

        after_count = len(split_dataset)
        log.info(
            "PeoplesSpeech '%s' duration filter kept %d/%d rows.",
            split_name,
            after_count,
            before_count,
        )

        return split_dataset

    @staticmethod
    def _require_hf_datasets() -> None:
        if load_dataset is None or Audio is None:
            raise RuntimeError(
                "PeoplesSpeechDataModule requires HuggingFace datasets. "
                "Install with `uv add datasets` or run `uv sync` after updating dependencies."
            ) from _DATASETS_IMPORT_ERROR

    def _build_wrapped_splits(self) -> Dict[str, PeoplesSpeechDataset]:
        self._require_hf_datasets()

        data_files = self._build_data_files()
        dataset_dict = load_dataset(  # type: ignore[misc]
            "parquet",
            data_files=data_files,
            cache_dir=self.cache_dir,
            download_mode="reuse_dataset_if_exists",
            keep_in_memory=False,
        )

        wrapped: Dict[str, PeoplesSpeechDataset] = {}
        for split_name in ("train", "validation", "test"):
            split_dataset = dataset_dict[split_name]
            split_dataset = self._apply_duration_filter(split_dataset, split_name)
            split_dataset = split_dataset.cast_column(
                self.audio_column,
                Audio(sampling_rate=self.target_sample_rate),  # type: ignore[operator]
            )

            wrapped[split_name] = PeoplesSpeechDataset(
                hf_split=split_dataset,
                split_name=split_name,
                audio_column=self.audio_column,
                id_column=self.id_column,
                text_column=self.text_column,
                max_length=self.max_audio_length,
                target_sample_rate=self.target_sample_rate,
                decode_error_policy=self.decode_error_policy,
            )

        return wrapped

    def setup(self, stage: Optional[str] = None) -> None:
        if self._wrapped_splits is None:
            self._wrapped_splits = self._build_wrapped_splits()

        if stage == "fit" or stage is None:
            self.train_dataset = self._wrapped_splits["train"]
            self.val_dataset = self._wrapped_splits["validation"]

        if stage == "test" or stage is None:
            self.test_dataset = self._wrapped_splits["test"]

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError(
                "Train dataset is not initialized. Call setup('fit') first."
            )

        kwargs: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.num_workers > 0 and self.persistent_workers,
            "collate_fn": partial(self.collate_fn, mode=self.collate_mode),
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError(
                "Val dataset is not initialized. Call setup('fit') first."
            )

        kwargs: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.num_workers > 0 and self.persistent_workers,
            "collate_fn": partial(self.collate_fn, mode=self.collate_mode),
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(self.val_dataset, **kwargs)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError(
                "Test dataset is not initialized. Call setup('test') first."
            )

        kwargs: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.num_workers > 0 and self.persistent_workers,
            "collate_fn": partial(self.collate_fn, mode=self.collate_mode),
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(self.test_dataset, **kwargs)

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]], mode: str = "pad") -> Dict[str, Any]:
        batch = [item for item in batch if not item.get("error", False)]
        if len(batch) == 0:
            raise RuntimeError("All items in batch failed to decode.")

        return collate_audio_batch(
            batch=batch,
            waveform_key="waveform",
            mode=mode,
        )
