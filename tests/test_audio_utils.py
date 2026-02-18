import numpy as np
import pytest
import torch

from src.data.audio_utils import DatasetResamplerCropper
from src.data.audio_utils import collate_audio_batch


def test_collate_audio_batch_pad_mode() -> None:
    batch = [
        {"waveform": torch.ones(1, 3), "index": 1},
        {"waveform": torch.ones(1, 5), "index": 2},
    ]

    collated = collate_audio_batch(batch=batch, mode="pad")

    assert collated["waveform"].shape == (2, 1, 5)
    assert collated["index"].shape == (2,)
    assert collated["index"].tolist() == [1, 2]


def test_collate_audio_batch_truncate_mode() -> None:
    batch = [
        {"waveform": torch.ones(1, 6), "meta": "a"},
        {"waveform": torch.ones(1, 4), "meta": "b"},
    ]

    collated = collate_audio_batch(batch=batch, mode="truncate")

    assert collated["waveform"].shape == (2, 1, 4)
    assert collated["meta"] == ["a", "b"]


def test_collate_audio_batch_include_exclude_keys() -> None:
    batch = [
        {"waveform": torch.ones(1, 3), "index": 1, "audio_name": "x"},
        {"waveform": torch.ones(1, 3), "index": 2, "audio_name": "y"},
    ]

    collated = collate_audio_batch(
        batch=batch,
        include_keys=["index", "audio_name"],
        exclude_keys=["audio_name"],
    )

    assert "index" in collated
    assert "audio_name" not in collated
    assert "waveform" in collated


def test_collate_audio_batch_raises_on_empty_batch() -> None:
    with pytest.raises(ValueError, match="Empty batch"):
        collate_audio_batch(batch=[])


def test_collate_audio_batch_raises_on_invalid_mode() -> None:
    batch = [{"waveform": torch.ones(1, 4)}]

    with pytest.raises(ValueError, match="Unknown mode"):
        collate_audio_batch(batch=batch, mode="invalid")


def test_dataset_resampler_cropper_same_sample_rate_crops_to_max_length() -> None:
    np.random.seed(0)
    cropper = DatasetResamplerCropper(target_sr=16000, max_length=100)
    waveform = torch.randn(300)

    output = cropper(waveform, source_sr=16000)

    assert output.shape[-1] == 100
    assert len(cropper.resamplers) == 0


def test_dataset_resampler_cropper_reuses_resampler_cache() -> None:
    np.random.seed(0)
    cropper = DatasetResamplerCropper(target_sr=16000, max_length=80)
    waveform = torch.randn(320)

    _ = cropper(waveform, source_sr=32000)
    _ = cropper(waveform, source_sr=32000)

    assert 32000 in cropper.resamplers
    assert len(cropper.resamplers) == 1
