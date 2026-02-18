import numpy as np
import pytest
import torch

from src.data.audioset_datamodule import AudioSetDataModule
from src.data.audio_utils import DatasetResamplerCropper
from src.data.audio_utils import collate_audio_batch
from src.models.audio_jepa_module import AudioJEPAModule


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


def test_audioset_collate_variable_length_and_model_step() -> None:
    torch.manual_seed(0)
    batch = [
        {
            "waveform": torch.randn(1, 320),
            "target": torch.zeros(527),
            "audio_name": "a",
            "index": 0,
        },
        {
            "waveform": torch.randn(1, 520),
            "target": torch.zeros(527),
            "audio_name": "b",
            "index": 1,
        },
        {
            "waveform": torch.randn(1, 410),
            "target": torch.zeros(527),
            "audio_name": "c",
            "index": 2,
        },
    ]

    collated = AudioSetDataModule.collate_fn(batch, mode="pad")
    assert collated["waveform"].shape == (3, 1, 520)

    module = AudioJEPAModule(
        optimizer=torch.optim.AdamW,
        net={
            "spectrogram": {
                "sample_rate": 16000,
                "n_fft": 256,
                "win_length": 256,
                "hop_length": 64,
                "n_mels": 32,
                "f_min": 0,
                "f_max": 8000,
            },
            "patch_embed": {
                "img_size": (16, 16),
                "patch_size": (4, 4),
                "in_chans": 1,
                "embed_dim": 32,
            },
            "masking": {
                "input_size": (16, 16),
                "patch_size": (4, 4),
                "mask_ratio": (0.5, 0.5),
            },
            "encoder": {
                "embed_dim": 32,
                "depth": 1,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "qkv_bias": True,
                "num_patches": 16,
                "img_size": (16, 16),
                "patch_size": (4, 4),
                "pos_embed_type": "rope",
            },
            "predictor": {
                "embed_dim": 32,
                "depth": 1,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "qkv_bias": True,
                "num_patches": 16,
                "img_size": (16, 16),
                "patch_size": (4, 4),
                "pos_embed_type": "rope",
            },
        },
        spectrogram_adjustment_mode="pad",
    )
    module.log = lambda *args, **kwargs: None
    loss = module.training_step({"waveform": collated["waveform"]}, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
