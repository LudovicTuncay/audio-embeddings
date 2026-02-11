import os
from functools import partial
from typing import Any, Dict, List, Optional, Union

import lightning as L
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from src.data.audio_utils import DatasetResamplerCropper, collate_audio_batch


class YT1BDataset(Dataset):
    """
    Dataset for YT-Temporal-1B data using Parquet metadata files.

    Args:
        parquet_path (str): Path to the parquet file containing metadata (must include 'file_path', 'video_id', 'duration_sec').
            If a 'sample_rate' column exists, it is used to avoid probing files for source sample rate.
        min_duration (Optional[float]): Minimum duration in seconds to include a file.
        max_duration (Optional[float]): Maximum duration in seconds to include a file.
        transform (Optional[callable]): Optional transform to apply to the waveform.
        max_length (Optional[int]): Maximum length of the waveform in samples (at target_sample_rate).
        target_sample_rate (int): Target sample rate for the waveform. Defaults to 16000.
        decode_window_sec (Optional[float]): Optional decode window length in seconds. If None,
            defaults to max_length / target_sample_rate (when max_length is set).
    """

    def __init__(
        self,
        parquet_path: str,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = 30.0,
        transform: Optional[Any] = None,
        max_length: Optional[int] = None,
        target_sample_rate: int = 16000,
        decode_window_sec: Optional[float] = None,
    ):
        print(f"Loading metadata from {parquet_path}...")
        self.transform = transform
        self.max_length = max_length
        self.target_sample_rate = target_sample_rate
        self.decode_window_sec = decode_window_sec

        # --- Metadata Loading ---
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found at: {parquet_path}")

        # Pyarrow is required for read_parquet
        try:
            df = pd.read_parquet(parquet_path)
        except ImportError:
            raise ImportError(
                "Please install pyarrow to read parquet files: `uv add pyarrow`"
            )

        required_cols = {"file_path", "video_id", "duration_sec"}
        if not required_cols.issubset(df.columns):
            # Check if we have compatible columns or raise error
            # Some datasets might use different names, strictly enforcing for now based on user prompt
            raise ValueError(
                f"Parquet file must contain columns: {required_cols}. Found: {df.columns.tolist()}"
            )

        if min_duration is not None and min_duration < 0:
            raise ValueError(f"min_duration must be >= 0, got {min_duration}")
        if max_duration is not None and max_duration < 0:
            raise ValueError(f"max_duration must be >= 0, got {max_duration}")
        if (
            min_duration is not None
            and max_duration is not None
            and min_duration > max_duration
        ):
            raise ValueError(
                "min_duration must be <= max_duration; "
                f"got min_duration={min_duration}, max_duration={max_duration}"
            )

        if min_duration is not None:
            df = df[df["duration_sec"] >= min_duration]
        if max_duration is not None:
            df = df[df["duration_sec"] <= max_duration]

        self.ids = df["video_id"].tolist()
        self.paths = df["file_path"].tolist()
        self.durations_sec = df["duration_sec"].tolist()
        if "sample_rate" in df.columns:
            sample_rates = pd.to_numeric(df["sample_rate"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            self.source_sample_rates: Optional[list[Optional[int]]] = [
                int(sr) if np.isfinite(sr) and sr > 0 else None for sr in sample_rates
            ]
        else:
            self.source_sample_rates = None
        self.length = len(self.ids)

        # --- Resampler ---
        # Uses the optimized class that caches resamplers
        self.resampler = DatasetResamplerCropper(
            target_sr=target_sample_rate, max_length=max_length
        )

        print(f"Dataset loaded. Length: {self.length:,}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        audio_path = self.paths[idx]
        audio_id = self.ids[idx]

        # Load waveform
        try:
            decode_window_sec = self.decode_window_sec
            if decode_window_sec is None and self.max_length is not None:
                decode_window_sec = self.max_length / self.target_sample_rate

            if decode_window_sec is None:
                waveform, sr = torchaudio.load(audio_path)
            else:
                duration_sec = float(self.durations_sec[idx])
                if duration_sec <= 0:
                    waveform, sr = torchaudio.load(audio_path)
                else:
                    source_sr: Optional[int]
                    if self.source_sample_rates is not None:
                        source_sr = self.source_sample_rates[idx]
                    else:
                        source_sr = None

                    if source_sr is None:
                        _, source_sr = torchaudio.load(
                            audio_path, frame_offset=0, num_frames=1
                        )

                    total_frames = max(1, int(duration_sec * source_sr))
                    max_decode_frames = max(1, int(decode_window_sec * source_sr))
                    decode_frames = min(max_decode_frames, total_frames)

                    if total_frames > decode_frames:
                        max_start = total_frames - decode_frames
                        frame_offset = int(np.random.randint(0, max_start + 1))
                    else:
                        frame_offset = 0

                    waveform, sr = torchaudio.load(
                        audio_path,
                        frame_offset=frame_offset,
                        num_frames=decode_frames,
                    )
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return a dummy silent waveform to prevent crash
            len_samples = (
                self.max_length if self.max_length else self.target_sample_rate
            )
            return {
                "waveform": torch.zeros(1, len_samples),
                "audio_name": audio_id,
                "index": idx,
                "error": True,
            }

        # Mix down to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample and crop
        waveform = self.resampler(waveform, source_sr=sr)

        # Ensure channel dim exists [1, T] if resampler stripped it or returned [T]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if self.transform:
            waveform = self.transform(waveform)

        return {
            "waveform": waveform,
            "audio_name": audio_id,
            "index": idx,
        }


class YT1BDataModule(L.LightningDataModule):
    """
    LightningDataModule for YT-Temporal-1B.

    Args:
        data_dir (str): Root directory for data.
        train_parquet (str): Filename of training parquet file.
        val_parquet (str): Filename of validation parquet file.
        test_parquet (str): Filename of test parquet file.
        batch_size (int): Batch size for dataloaders.
        num_workers (int): Number of workers for dataloaders.
        pin_memory (bool): Whether to pin memory in dataloaders.
        max_audio_length_sec (Optional[float]): Maximum audio length in seconds.
        min_duration_sec (Optional[float]): Minimum audio duration in seconds to filter.
        max_duration_sec (Optional[float]): Maximum audio duration in seconds to filter.
        target_sample_rate (int): Target sample rate.
        collate_mode (str): 'pad' or 'truncate'.
        decode_window_sec (Optional[float]): Optional decode window length in seconds. If None,
            defaults to max_audio_length_sec.
    """

    def __init__(
        self,
        data_dir: str = "data/YT-Temporal-1B",
        train_parquet: str = "train_metadata.parquet",
        val_parquet: str = "val_metadata.parquet",
        test_parquet: str = "val_metadata.parquet",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_audio_length_sec: Optional[float] = 10.0,
        min_duration_sec: Optional[float] = None,
        max_duration_sec: Optional[float] = 30.0,
        target_sample_rate: int = 16000,
        collate_mode: str = "pad",
        decode_window_sec: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.train_parquet_path = os.path.join(data_dir, train_parquet)
        self.val_parquet_path = os.path.join(data_dir, val_parquet)
        self.test_parquet_path = os.path.join(data_dir, test_parquet)

        if max_audio_length_sec is not None:
            self.max_audio_length = int(max_audio_length_sec * target_sample_rate)
        else:
            self.max_audio_length = None

        self.train_dataset: Optional[YT1BDataset] = None
        self.val_dataset: Optional[YT1BDataset] = None
        self.test_dataset: Optional[YT1BDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            if os.path.exists(self.train_parquet_path):
                self.train_dataset = YT1BDataset(
                    self.train_parquet_path,
                    min_duration=self.hparams["min_duration_sec"],
                    max_duration=self.hparams["max_duration_sec"],
                    max_length=self.max_audio_length,
                    target_sample_rate=self.hparams["target_sample_rate"],
                    decode_window_sec=self.hparams["decode_window_sec"],
                )

            if os.path.exists(self.val_parquet_path):
                self.val_dataset = YT1BDataset(
                    self.val_parquet_path,
                    min_duration=self.hparams["min_duration_sec"],
                    max_duration=self.hparams["max_duration_sec"],
                    max_length=self.max_audio_length,
                    target_sample_rate=self.hparams["target_sample_rate"],
                    decode_window_sec=self.hparams["decode_window_sec"],
                )

        if stage == "test":
            if os.path.exists(self.test_parquet_path):
                self.test_dataset = YT1BDataset(
                    self.test_parquet_path,
                    min_duration=self.hparams["min_duration_sec"],
                    max_duration=self.hparams["max_duration_sec"],
                    max_length=self.max_audio_length,
                    target_sample_rate=self.hparams["target_sample_rate"],
                    decode_window_sec=self.hparams["decode_window_sec"],
                )

    def train_dataloader(self) -> DataLoader:
        if not self.train_dataset:
            raise RuntimeError(
                f"Train dataset not initialized. File not found: {self.train_parquet_path}"
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            persistent_workers=self.hparams["num_workers"] > 0,
            collate_fn=partial(self.collate_fn, mode=self.hparams["collate_mode"]),
        )

    def val_dataloader(self) -> DataLoader:
        if not self.val_dataset:
            # Often validation sets are missing in large scale pretraining or we use a subset of train
            # For now, raise strict error or return empty list (lightning supports empty list for no val)
            # Raising error is safer to debug configuration issues.
            raise RuntimeError(
                f"Val dataset not initialized. File not found: {self.val_parquet_path}"
            )

        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            persistent_workers=self.hparams["num_workers"] > 0,
            collate_fn=partial(self.collate_fn, mode=self.hparams["collate_mode"]),
        )

    def test_dataloader(self) -> DataLoader:
        if not self.test_dataset:
            raise RuntimeError(
                f"Test dataset not initialized. File not found: {self.test_parquet_path}"
            )

        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            collate_fn=partial(self.collate_fn, mode=self.hparams["collate_mode"]),
        )

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]], mode: str = "pad") -> Dict[str, Any]:
        # Filter out errors
        batch = [x for x in batch if not x.get("error", False)]
        if len(batch) == 0:
            raise RuntimeError("All items in batch failed to load.")

        return collate_audio_batch(
            batch=batch,
            waveform_key="waveform",
            mode=mode,
        )
