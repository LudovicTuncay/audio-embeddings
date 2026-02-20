from pathlib import Path
from typing import Any, Dict

import pytest
import torch

import src.data.peoples_speech_datamodule as ps_module
from src.data.peoples_speech_datamodule import (
    PeoplesSpeechDataModule,
    PeoplesSpeechDataset,
)


class FakeAudio:
    def __init__(self, sampling_rate: int):
        self.sampling_rate = sampling_rate


class FakeHFDataset:
    def __init__(self, rows: list[Dict[str, Any]]):
        self.rows = list(rows)
        self.cast_calls: list[tuple[str, int]] = []

    @property
    def column_names(self) -> list[str]:
        if len(self.rows) == 0:
            return []
        return list(self.rows[0].keys())

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]

    def filter(self, fn: Any, desc: str | None = None) -> "FakeHFDataset":
        del desc
        return FakeHFDataset([row for row in self.rows if fn(row)])

    def cast_column(self, column: str, feature: Any) -> "FakeHFDataset":
        self.cast_calls.append((column, int(feature.sampling_rate)))
        return self


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_split_discovery_uses_configured_patterns(tmp_path: Path) -> None:
    root = tmp_path / "peoples_speech" / "clean"
    _touch(root / "train.parquet")
    _touch(root / "valid.parquet")
    _touch(root / "test.parquet")

    dm = PeoplesSpeechDataModule(
        data_root=str(tmp_path / "peoples_speech"),
        subset="clean",
        split_file_patterns={
            "train": ["train*.parquet"],
            "validation": ["valid*.parquet"],
            "test": ["test*.parquet"],
        },
        num_workers=0,
        pin_memory=False,
    )

    data_files = dm._build_data_files()

    assert len(data_files["train"]) == 1
    assert len(data_files["validation"]) == 1
    assert len(data_files["test"]) == 1
    assert data_files["train"][0].endswith("train.parquet")


def test_setup_applies_duration_filter_and_audio_cast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "peoples_speech" / "clean"
    _touch(root / "train.parquet")
    _touch(root / "valid.parquet")
    _touch(root / "test.parquet")

    train_rows = [
        {
            "id": "drop-short",
            "duration_ms": 9000,
            "audio": {"array": [0.0] * 32, "sampling_rate": 16000},
            "text": "a",
        },
        {
            "id": "keep-min",
            "duration_ms": 10000,
            "audio": {"array": [0.1] * 32, "sampling_rate": 16000},
            "text": "b",
        },
        {
            "id": "keep-mid",
            "duration_ms": 25000,
            "audio": {"array": [0.2] * 32, "sampling_rate": 16000},
            "text": "c",
        },
        {
            "id": "drop-long",
            "duration_ms": 35000,
            "audio": {"array": [0.3] * 32, "sampling_rate": 16000},
            "text": "d",
        },
    ]

    dataset_dict: Dict[str, FakeHFDataset] = {
        "train": FakeHFDataset(train_rows),
        "validation": FakeHFDataset(
            [
                {
                    "id": "val",
                    "duration_ms": 12000,
                    "audio": {"array": [0.1] * 32, "sampling_rate": 16000},
                    "text": "val",
                }
            ]
        ),
        "test": FakeHFDataset(
            [
                {
                    "id": "test",
                    "duration_ms": 12000,
                    "audio": {"array": [0.1] * 32, "sampling_rate": 16000},
                    "text": "test",
                }
            ]
        ),
    }

    def fake_load_dataset(*args: Any, **kwargs: Any) -> Dict[str, FakeHFDataset]:
        assert args[0] == "parquet"
        assert kwargs["keep_in_memory"] is False
        return dataset_dict

    monkeypatch.setattr(ps_module, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(ps_module, "Audio", FakeAudio)

    dm = PeoplesSpeechDataModule(
        data_root=str(tmp_path / "peoples_speech"),
        subset="clean",
        min_duration_sec=10.0,
        max_duration_sec=30.0,
        target_sample_rate=16000,
        num_workers=0,
        pin_memory=False,
    )

    dm.setup("fit")

    assert dm.train_dataset is not None
    assert len(dm.train_dataset) == 2

    sample = dm.train_dataset[0]
    assert sample["waveform"].dtype == torch.float32
    assert sample["waveform"].ndim == 2
    assert sample["waveform"].shape[0] == 1
    assert sample["audio_name"] in {"keep-min", "keep-mid"}


def test_dataset_returns_waveform_key_with_mono_shape() -> None:
    hf_rows = [
        {
            "id": "abc",
            "audio": {
                # [T, C] style stereo input
                "array": [[0.1, -0.1] for _ in range(32)],
                "sampling_rate": 16000,
            },
            "text": "hello",
        }
    ]

    ds = PeoplesSpeechDataset(
        hf_split=FakeHFDataset(hf_rows),
        split_name="train",
        audio_column="audio",
        id_column="id",
        text_column="text",
        max_length=16,
        target_sample_rate=16000,
        decode_error_policy="skip",
    )

    sample = ds[0]

    assert "waveform" in sample
    assert sample["waveform"].shape == (1, 16)
    assert sample["waveform"].dtype == torch.float32
    assert sample["audio_name"] == "abc"


def test_collate_skips_decode_errors() -> None:
    batch = [
        {"waveform": torch.ones(1, 5), "audio_name": "ok", "index": 0},
        {"audio_name": "bad", "index": 1, "error": True},
    ]

    collated = PeoplesSpeechDataModule.collate_fn(batch, mode="pad")
    assert collated["waveform"].shape == (1, 1, 5)

    with pytest.raises(RuntimeError, match="All items in batch failed"):
        PeoplesSpeechDataModule.collate_fn(
            [
                {"audio_name": "bad-1", "index": 1, "error": True},
                {"audio_name": "bad-2", "index": 2, "error": True},
            ]
        )
