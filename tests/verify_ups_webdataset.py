import tarfile
import tempfile
import sys
import os
from pathlib import Path

import numpy as np
import soundfile as sf

# Ensure local `src` package is importable when run as a script.
sys.path.append(os.path.abspath("."))

from src.data.ups_webdataset_datamodule import UPSWebDatasetDataModule


def _write_test_wav(path: Path, sample_rate: int, duration_sec: float) -> None:
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0.0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
    waveform = 0.1 * np.sin(2.0 * np.pi * 220.0 * t)
    sf.write(path, waveform, sample_rate, format="WAV", subtype="PCM_16")


def _create_test_shard(shard_path: Path, shard_idx: int, sample_rate: int) -> None:
    with tempfile.TemporaryDirectory() as local_tmp:
        tmp_dir = Path(local_tmp)
        wav_paths: list[Path] = []
        txt_paths: list[Path] = []

        for item_idx, duration_sec in enumerate((0.4, 1.3)):
            stem = f"item_{shard_idx:06d}_{item_idx:02d}"
            wav_path = tmp_dir / f"{stem}.wav"
            txt_path = tmp_dir / f"{stem}.txt"

            _write_test_wav(
                wav_path, sample_rate=sample_rate, duration_sec=duration_sec
            )
            txt_path.write_text("synthetic sidecar", encoding="utf-8")

            wav_paths.append(wav_path)
            txt_paths.append(txt_path)

        with tarfile.open(shard_path, "w") as tar:
            for wav_path, txt_path in zip(wav_paths, txt_paths):
                # UPS-style key: audio payload is addressed via `.mp3` extension.
                tar.add(wav_path, arcname=wav_path.with_suffix(".mp3").name)
                tar.add(txt_path, arcname=txt_path.name)


def verify_ups_webdataset() -> None:
    target_sample_rate = 16000
    chunk_sec = 1.0
    expected_t = int(target_sample_rate * chunk_sec)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        audio_dir = tmp_path / "audio"
        audio2_dir = tmp_path / "audio2"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio2_dir.mkdir(parents=True, exist_ok=True)

        for shard_num in range(1, 7):
            if shard_num <= 4:
                shard_dir = audio_dir
            else:
                shard_dir = audio2_dir
            shard_path = shard_dir / f"{shard_num:06d}.tar"
            _create_test_shard(
                shard_path, shard_idx=shard_num, sample_rate=target_sample_rate
            )

        dm = UPSWebDatasetDataModule(
            shard_globs=[str(audio_dir / "*.tar"), str(audio2_dir / "*.tar")],
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            target_sample_rate=target_sample_rate,
            chunk_sec=chunk_sec,
            collate_mode="pad",
            val_num_shards=1,
            test_num_shards=1,
            split_seed=123,
            train_shardshuffle=0,
            train_sampleshuffle=0,
            eval_sampleshuffle=0,
            persistent_workers=False,
            drop_last=True,
        )

        dm.setup()

        assert len(dm.train_shards) > 0, "Train split should be non-empty."
        assert len(dm.val_shards) > 0, "Val split should be non-empty."
        assert len(dm.test_shards) > 0, "Test split should be non-empty."

        train_set = set(dm.train_shards)
        val_set = set(dm.val_shards)
        test_set = set(dm.test_shards)

        assert train_set.isdisjoint(val_set), "Train and val splits overlap."
        assert train_set.isdisjoint(test_set), "Train and test splits overlap."
        assert val_set.isdisjoint(test_set), "Val and test splits overlap."

        train_batch = next(iter(dm.train_dataloader()))
        val_batch = next(iter(dm.val_dataloader()))
        test_batch = next(iter(dm.test_dataloader()))

        for batch_name, batch in (
            ("train", train_batch),
            ("val", val_batch),
            ("test", test_batch),
        ):
            waveform = batch["waveform"]
            assert waveform.ndim == 3, f"{batch_name}: waveform must be [B, C, T]"
            assert waveform.shape[1] == 1, f"{batch_name}: channel dim must be 1"
            assert waveform.shape[2] == expected_t, (
                f"{batch_name}: expected T={expected_t}, got {waveform.shape[2]}"
            )

    print("UPS WebDataset verification successful.")


if __name__ == "__main__":
    verify_ups_webdataset()
