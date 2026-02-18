from pathlib import Path
import subprocess
import sys

import pytest


def _tiny_model_overrides() -> list[str]:
    """Overrides to keep fast-dev-run smoke tests lightweight on CPU."""
    return [
        "model.net.spectrogram.n_fft=256",
        "model.net.spectrogram.win_length_ms=16",
        "model.net.spectrogram.hop_length_ms=4",
        "model.net.spectrogram.n_mels=32",
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


def _run_train_command(
    tmp_path: Path, extra_overrides: list[str]
) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "src/train.py",
        "data=mock_audioset",
        "trainer=cpu",
        "logger=[]",
        "callbacks=none",
        "test=false",
        "extras.enforce_tags=false",
        "extras.print_config=false",
        "+trainer.fast_dev_run=true",
        "data.batch_size=2",
        "data.max_audio_length_sec=0.2",
        f"hydra.run.dir={tmp_path}",
    ]
    command.extend(_tiny_model_overrides())
    command.extend(extra_overrides)
    return subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def test_train_fast_dev_run_cpu(tmp_path: Path) -> None:
    """Train entrypoint should complete a fast-dev-run on CPU."""
    result = _run_train_command(tmp_path=tmp_path, extra_overrides=[])

    assert result.returncode == 0, result.stderr
    assert "Starting training!" in result.stdout


@pytest.mark.slow
def test_train_fast_dev_run_checkpoint_dir_created(tmp_path: Path) -> None:
    """When checkpoint callback is enabled, the run should create a checkpoint folder."""
    result = _run_train_command(
        tmp_path=tmp_path,
        extra_overrides=[
            "callbacks=default",
            "callbacks.device_stats=null",
            "callbacks.visualization=null",
            "callbacks.safetensors=null",
            "callbacks.model_checkpoint.save_last=true",
            "callbacks.model_checkpoint.save_top_k=0",
        ],
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "checkpoints").exists()
