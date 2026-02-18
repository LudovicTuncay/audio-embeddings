import functools
from typing import Any
from types import SimpleNamespace

import torch

from src.models.audio_jepa_module import AudioJEPAModule
from src.models.best_rq2_module import BestRQ2Module
from src.models.rqa_jepa_module import RQAJEPAModule


def _make_tiny_net_config() -> dict[str, Any]:
    return {
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


def test_rqa_jepa_training_step_teacher_targets_returns_finite_scalar_loss() -> None:
    torch.manual_seed(0)
    module = RQAJEPAModule(
        optimizer=functools.partial(torch.optim.AdamW, lr=1e-3),
        net=_make_tiny_net_config(),
        spectrogram_adjustment_mode="truncate",
        rq_input_type="teacher",
        codebook_dim=8,
        vocab_size=64,
    )
    module.log = lambda *args, **kwargs: None
    module.current_ema_decay = 0.99

    batch = {"waveform": torch.randn(2, 1, 640)}
    loss = module.training_step(batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_rqa_jepa_training_step_spectrogram_targets_returns_finite_scalar_loss() -> (
    None
):
    torch.manual_seed(0)
    module = RQAJEPAModule(
        optimizer=functools.partial(torch.optim.AdamW, lr=1e-3),
        net=_make_tiny_net_config(),
        spectrogram_adjustment_mode="truncate",
        rq_input_type="spectrogram",
        codebook_dim=8,
        vocab_size=64,
    )
    module.log = lambda *args, **kwargs: None
    module.current_ema_decay = 0.99

    batch = {"waveform": torch.randn(2, 1, 640)}
    loss = module.training_step(batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_audio_jepa_configure_optimizers_wires_lambda_scheduler() -> None:
    module = AudioJEPAModule(
        optimizer=functools.partial(torch.optim.SGD, lr=1.0),
        net=_make_tiny_net_config(),
        warmup_pct=0.1,
        final_lr_ratio=0.05,
    )
    module.trainer = SimpleNamespace(max_steps=20, estimated_stepping_batches=20)

    optim_conf = module.configure_optimizers()
    optimizer = optim_conf["optimizer"]
    scheduler = optim_conf["lr_scheduler"]["scheduler"]

    for _ in range(20):
        optimizer.step()
        scheduler.step()

    lr = scheduler.get_last_lr()[0]
    assert 0.05 <= lr <= 1.0
