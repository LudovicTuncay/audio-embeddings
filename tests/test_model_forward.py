import functools
from typing import Any

import torch

from src.models.audio_jepa_module import AudioJEPAModule
from src.models.best_rq2_module import BestRQ2Module


def _make_tiny_net_config() -> dict[str, Any]:
    return {
        "spectrogram": {
            "sample_rate": 16000,
            "n_fft": 64,
            "win_length": 64,
            "hop_length": 16,
            "n_mels": 16,
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
    }


def test_audio_jepa_training_step_returns_finite_scalar_loss() -> None:
    torch.manual_seed(0)
    module = AudioJEPAModule(
        optimizer=functools.partial(torch.optim.AdamW, lr=1e-3),
        net=_make_tiny_net_config(),
        spectrogram_adjustment_mode="truncate",
    )
    module.log = lambda *args, **kwargs: None

    batch = {"waveform": torch.randn(2, 1, 640)}
    loss = module.training_step(batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_best_rq2_training_step_returns_finite_scalar_loss() -> None:
    torch.manual_seed(0)
    module = BestRQ2Module(
        optimizer=functools.partial(torch.optim.AdamW, lr=1e-3),
        net=_make_tiny_net_config(),
        spectrogram_adjustment_mode="truncate",
        codebook_dim=8,
        vocab_size=64,
    )
    module.log = lambda *args, **kwargs: None

    batch = {"waveform": torch.randn(2, 1, 640)}
    loss = module.training_step(batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
