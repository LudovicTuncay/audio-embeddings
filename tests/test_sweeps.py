from pathlib import Path
import subprocess
import sys

import pytest


def _tiny_model_overrides() -> list[str]:
    return [
        "model.net.spectrogram.n_fft=64",
        "model.net.spectrogram.win_length_ms=4",
        "model.net.spectrogram.hop_length_ms=2",
        "model.net.spectrogram.n_mels=16",
        "model.net.patch_embed.img_size=[16,16]",
        "model.net.patch_embed.patch_size=[4,4]",
        "model.net.patch_embed.embed_dim=32",
        "model.net.masking.input_size=[16,16]",
        "model.net.masking.patch_size=[4,4]",
        "model.net.encoder.embed_dim=32",
        "model.net.encoder.depth=1",
        "model.net.encoder.num_heads=4",
        "model.net.encoder.mlp_ratio=2.0",
        "model.net.encoder.num_patches=16",
        "model.net.encoder.pos_embed_type=rope",
        "model.net.predictor.embed_dim=32",
        "model.net.predictor.depth=1",
        "model.net.predictor.num_heads=4",
        "model.net.predictor.mlp_ratio=2.0",
        "model.net.predictor.num_patches=16",
        "model.net.predictor.pos_embed_type=rope",
    ]


@pytest.mark.slow
@pytest.mark.integration
def test_hydra_multirun_lr_sweep_with_mock_data(tmp_path: Path) -> None:
    """A minimal multirun sweep should execute with current repo configs."""
    repo_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "src/train.py",
        "-m",
        f"hydra.sweep.dir={tmp_path}",
        "model.optimizer.lr=0.001,0.002",
        "data=mock_audioset",
        "trainer=cpu",
        "logger=[]",
        "callbacks=none",
        "test=false",
        "extras.enforce_tags=false",
        "extras.print_config=false",
        "++trainer.fast_dev_run=true",
        "data.batch_size=2",
        "data.max_audio_length_sec=0.2",
    ]
    command.extend(_tiny_model_overrides())

    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
