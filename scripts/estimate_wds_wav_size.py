#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torchaudio
from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate storage required for a WAV-only WebDataset generated from a parquet manifest."
        )
    )
    parser.add_argument(
        "parquet_path",
        type=Path,
        help="Path to parquet file (must include audio path column).",
    )
    parser.add_argument(
        "--audio-column",
        type=str,
        default="file_path",
        help="Column name containing source audio file paths.",
    )
    parser.add_argument(
        "--duration-column",
        type=str,
        default="duration_sec",
        help="Column name containing durations in seconds.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Parallel worker count used to probe missing durations.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target WAV sample rate.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Target WAV channel count.",
    )
    parser.add_argument(
        "--bits-per-sample",
        type=int,
        default=16,
        choices=[8, 16, 24, 32],
        help="Target WAV PCM bit depth.",
    )
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=1.0,
        help="Assumed max size per .tar shard in GB (decimal) for trailer overhead estimate.",
    )
    parser.add_argument(
        "--no-probe-missing",
        action="store_true",
        help="Fail if duration is missing/invalid instead of probing audio headers.",
    )
    return parser.parse_args()


def format_bytes(n_bytes: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    value = float(n_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:,.2f} {unit}"
        value /= 1024.0
    return f"{value:,.2f} PiB"


def estimate_shard_count(total_bytes: int, shard_size_gb: float) -> int:
    shard_size_bytes = max(1, int(shard_size_gb * 1_000_000_000))
    return max(1, math.ceil(total_bytes / shard_size_bytes))


def parse_duration_value(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return float("nan")
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return float("nan")


def probe_duration_sec(audio_path: str) -> float:
    try:
        info_fn = getattr(torchaudio, "info", None)
        if info_fn is None:
            return float("nan")
        info = info_fn(audio_path)
        if info.sample_rate <= 0:
            return float("nan")
        return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        return float("nan")


def probe_file_size_bytes(audio_path: str) -> int:
    try:
        return Path(audio_path).stat().st_size
    except Exception:
        return -1


def probe_durations_parallel(paths: list[str], workers: int) -> list[float]:
    if not paths:
        return []

    results: list[float] = [float("nan")] * len(paths)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_idx: dict[Future[float], int] = {
            executor.submit(probe_duration_sec, path): idx
            for idx, path in enumerate(paths)
        }

        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task_id = progress.add_task("Probing missing durations", total=len(paths))
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = float("nan")
                progress.advance(task_id, 1)

    return results


def probe_file_sizes_parallel(paths: list[str], workers: int) -> list[int]:
    if not paths:
        return []

    results: list[int] = [0] * len(paths)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_idx: dict[Future[int], int] = {
            executor.submit(probe_file_size_bytes, path): idx
            for idx, path in enumerate(paths)
        }

        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task_id = progress.add_task("Probing current file sizes", total=len(paths))
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = -1
                progress.advance(task_id, 1)

    return results


def main() -> None:
    args = parse_args()

    if not args.parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {args.parquet_path}")

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    df = pd.read_parquet(args.parquet_path)

    if args.audio_column not in df.columns:
        raise ValueError(f"Missing audio column '{args.audio_column}' in parquet.")
    if args.duration_column not in df.columns:
        raise ValueError(
            f"Missing duration column '{args.duration_column}' in parquet. "
            "Either add it or adapt the script."
        )

    durations = np.array(
        [parse_duration_value(value) for value in df[args.duration_column].tolist()],
        dtype=np.float64,
    )
    paths = [str(path) for path in df[args.audio_column].tolist()]
    n_rows = len(paths)

    unique_paths = list(dict.fromkeys(paths))
    unique_sizes = probe_file_sizes_parallel(unique_paths, workers=args.workers)
    size_by_path = dict(zip(unique_paths, unique_sizes))
    row_sizes = np.array([size_by_path[path] for path in paths], dtype=np.int64)
    current_row_known_mask = row_sizes >= 0
    current_unique_known_mask = np.array(unique_sizes, dtype=np.int64) >= 0

    current_rows_total_bytes = int(row_sizes[current_row_known_mask].sum())
    current_unique_total_bytes = int(
        np.array(unique_sizes, dtype=np.int64)[current_unique_known_mask].sum()
    )
    current_rows_missing = int((~current_row_known_mask).sum())
    current_unique_missing = int((~current_unique_known_mask).sum())

    invalid_mask = ~np.isfinite(durations) | (durations <= 0.0)
    n_missing = int(invalid_mask.sum())

    if n_missing > 0:
        if args.no_probe_missing:
            raise ValueError(
                f"Found {n_missing} rows with missing/invalid durations and --no-probe-missing was set."
            )

        probe_indices = np.where(invalid_mask)[0].tolist()
        probe_paths = [paths[i] for i in probe_indices]
        probed = probe_durations_parallel(probe_paths, workers=args.workers)
        for i, duration in zip(probe_indices, probed, strict=True):
            durations[i] = duration

    unresolved_mask = ~np.isfinite(durations) | (durations <= 0.0)
    n_unresolved = int(unresolved_mask.sum())

    valid_durations = durations[~unresolved_mask]
    n_valid = int(valid_durations.shape[0])
    if n_valid == 0:
        raise RuntimeError("No valid durations available; cannot estimate size.")

    bytes_per_sample = args.bits_per_sample // 8
    bytes_per_frame = args.channels * bytes_per_sample

    frames = np.rint(valid_durations * args.sample_rate).astype(np.int64)
    wav_file_bytes = 44 + frames * bytes_per_frame
    wav_total_bytes = int(wav_file_bytes.sum())

    padded_wav_data = ((wav_file_bytes + 511) // 512) * 512
    tar_entry_overhead = 512 * n_valid
    tar_data_bytes = int(padded_wav_data.sum())

    estimated_shards = estimate_shard_count(wav_total_bytes, args.shard_size_gb)
    tar_trailer_overhead = 1024 * estimated_shards

    wds_tar_total_bytes = tar_entry_overhead + tar_data_bytes + tar_trailer_overhead

    total_duration_sec = float(valid_durations.sum())
    total_hours = total_duration_sec / 3600.0

    print("=== WebDataset WAV Size Estimate ===")
    print(f"Parquet: {args.parquet_path}")
    print(f"Rows total: {n_rows:,}")
    print(f"Rows valid: {n_valid:,}")
    print(f"Rows missing/invalid duration initially: {n_missing:,}")
    print(f"Rows unresolved after probe: {n_unresolved:,}")
    print()
    print("Current dataset volume (before conversion):")
    print(
        "- Referenced rows total bytes: "
        f"{format_bytes(current_rows_total_bytes)} "
        f"(missing rows: {current_rows_missing:,})"
    )
    print(
        "- Unique source files total bytes: "
        f"{format_bytes(current_unique_total_bytes)} "
        f"(missing files: {current_unique_missing:,})"
    )
    print()
    print(
        f"Target WAV format: {args.sample_rate} Hz, {args.channels} ch, "
        f"PCM {args.bits_per_sample}-bit"
    )
    print(f"Total audio duration: {total_hours:,.2f} h")
    print()
    print(f"Estimated WAV bytes (sum of .wav files): {format_bytes(wav_total_bytes)}")
    print(
        f"Estimated WDS TAR bytes (wav-only):      {format_bytes(wds_tar_total_bytes)}"
    )
    print(
        f"Average WAV per sample:                  {format_bytes(wav_total_bytes / n_valid)}"
    )
    print(
        f"Average WDS per sample:                  {format_bytes(wds_tar_total_bytes / n_valid)}"
    )
    print()
    print(f"Assumed shard size: {args.shard_size_gb} GB")
    print(f"Estimated shard count: {estimated_shards:,}")
    print()
    print("Shard count quick table (for WAV payload):")
    for shard_size_gb in [1.0, 2.0, 5.0, 10.0]:
        shard_count = estimate_shard_count(wav_total_bytes, shard_size_gb)
        print(f"- {shard_size_gb:>4.1f} GB -> {shard_count:,} shards")


if __name__ == "__main__":
    main()
