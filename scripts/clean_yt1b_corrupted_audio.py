import argparse
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import statistics
import time

import pandas as pd
from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from torch.utils.data import DataLoader

from src.data.yt1b_datamodule import YT1BDataModule
from src.data.yt1b_datamodule import YT1BDataset


def identity_collate(batch: list[dict]) -> list[dict]:
    return batch


@dataclass
class SplitScanStats:
    processed_samples: int
    error_samples: int
    unique_bad_paths: int
    num_batches: int
    elapsed_sec: float
    mean_batch_sec: float
    p50_batch_sec: float
    p90_batch_sec: float
    p99_batch_sec: float

    @property
    def samples_per_sec(self) -> float:
        if self.elapsed_sec <= 0:
            return 0.0
        return self.processed_samples / self.elapsed_sec

    @property
    def error_rate(self) -> float:
        if self.processed_samples == 0:
            return 0.0
        return self.error_samples / self.processed_samples


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0

    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]

    q_clamped = max(0.0, min(1.0, q))
    idx = q_clamped * (len(sorted_vals) - 1)
    low = int(idx)
    high = min(low + 1, len(sorted_vals) - 1)
    weight = idx - low
    return sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight


def scan_split_for_failures(
    split_name: str,
    dataset: YT1BDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[set[str], SplitScanStats]:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=identity_collate,
    )

    bad_paths: set[str] = set()
    batch_latencies: list[float] = []
    processed_samples = 0
    error_samples = 0
    num_batches = 0
    start_time = time.perf_counter()

    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task(f"Scanning {split_name}", total=len(dataset))

        dataloader_iter = iter(dataloader)
        while True:
            batch_start = time.perf_counter()
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break

            fetch_and_process_sec = time.perf_counter() - batch_start
            for sample in batch:
                processed_samples += 1
                if sample.get("error", False):
                    error_samples += 1
                    sample_index = int(sample["index"])
                    bad_paths.add(dataset.paths[sample_index])
            num_batches += 1
            batch_latencies.append(fetch_and_process_sec)
            progress.advance(task_id, len(batch))

    elapsed_sec = time.perf_counter() - start_time
    if batch_latencies:
        mean_batch_sec = statistics.fmean(batch_latencies)
        p50_batch_sec = percentile(batch_latencies, 0.50)
        p90_batch_sec = percentile(batch_latencies, 0.90)
        p99_batch_sec = percentile(batch_latencies, 0.99)
    else:
        mean_batch_sec = 0.0
        p50_batch_sec = 0.0
        p90_batch_sec = 0.0
        p99_batch_sec = 0.0

    stats = SplitScanStats(
        processed_samples=processed_samples,
        error_samples=error_samples,
        unique_bad_paths=len(bad_paths),
        num_batches=num_batches,
        elapsed_sec=elapsed_sec,
        mean_batch_sec=mean_batch_sec,
        p50_batch_sec=p50_batch_sec,
        p90_batch_sec=p90_batch_sec,
        p99_batch_sec=p99_batch_sec,
    )

    return bad_paths, stats


def clean_parquet_file(
    parquet_path: str, bad_paths: Iterable[str], dry_run: bool
) -> int:
    bad_paths_set = set(bad_paths)
    if not bad_paths_set:
        return 0

    df = pd.read_parquet(parquet_path)
    if "file_path" not in df.columns:
        raise ValueError(
            f"Parquet file must contain 'file_path' column: {parquet_path}"
        )

    bad_mask = df["file_path"].isin(list(bad_paths_set))
    removed = int(bad_mask.sum())

    if removed > 0 and not dry_run:
        cleaned_df = df.loc[~bad_mask].reset_index(drop=True)
        cleaned_df.to_parquet(parquet_path, index=False)

    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan YT-Temporal-1B train/val/test splits with the existing dataloader, "
            "detect decode failures, and remove failing files from parquet metadata."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/lustre/fswork/projects/rech/ojz/umz91bs/audio-embeddings/data/YT-Temporal-1B/",
        help="Root directory containing the parquet metadata files.",
    )
    parser.add_argument(
        "--train-parquet",
        type=str,
        default="train_metadata.parquet",
        help="Train parquet filename under --data-dir.",
    )
    parser.add_argument(
        "--val-parquet",
        type=str,
        default="val_metadata.parquet",
        help="Validation parquet filename under --data-dir.",
    )
    parser.add_argument(
        "--test-parquet",
        type=str,
        default="val_metadata.parquet",
        help="Test parquet filename under --data-dir.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for scanning.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=24,
        help="Number of dataloader workers (CPU cores).",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pin_memory for dataloaders.",
    )
    parser.add_argument(
        "--max-audio-length-sec",
        type=float,
        default=10.0,
        help="Maximum waveform duration in seconds while scanning.",
    )
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=None,
        help="Optional minimum duration filter (same as datamodule).",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=16000,
        help="Target sampling rate used by the dataset resampler.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report removals without modifying parquet files.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print detailed throughput and latency metrics per split.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    datamodule = YT1BDataModule(
        data_dir=args.data_dir,
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        test_parquet=args.test_parquet,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        max_audio_length_sec=args.max_audio_length_sec,
        min_duration_sec=args.min_duration_sec,
        target_sample_rate=args.target_sample_rate,
    )

    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    split_specs = [
        ("train", datamodule.train_dataset, datamodule.train_parquet_path),
        ("val", datamodule.val_dataset, datamodule.val_parquet_path),
        ("test", datamodule.test_dataset, datamodule.test_parquet_path),
    ]

    bad_paths_by_parquet: dict[str, set[str]] = defaultdict(set)
    bad_counts_by_split: dict[str, int] = {}
    stats_by_split: dict[str, SplitScanStats] = {}

    for split_name, dataset, parquet_path in split_specs:
        if dataset is None:
            print(f"Skipping {split_name}: parquet not found at {parquet_path}")
            continue

        bad_paths, stats = scan_split_for_failures(
            split_name=split_name,
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        bad_counts_by_split[split_name] = len(bad_paths)
        stats_by_split[split_name] = stats
        bad_paths_by_parquet[parquet_path].update(bad_paths)

    print("\nFailure counts by split:")
    for split_name in ["train", "val", "test"]:
        if split_name in bad_counts_by_split:
            print(f"- {split_name}: {bad_counts_by_split[split_name]}")

    if args.profile:
        print("\nProfile report:")
        for split_name in ["train", "val", "test"]:
            if split_name not in stats_by_split:
                continue

            stats = stats_by_split[split_name]
            print(
                f"- {split_name}: {stats.processed_samples} samples in "
                f"{stats.elapsed_sec:.1f}s ({stats.samples_per_sec:.2f} samples/s), "
                f"errors={stats.error_samples} ({100.0 * stats.error_rate:.2f}%), "
                f"unique_bad={stats.unique_bad_paths}, batches={stats.num_batches}"
            )
            print(
                f"  batch latency (s): mean={stats.mean_batch_sec:.4f}, "
                f"p50={stats.p50_batch_sec:.4f}, p90={stats.p90_batch_sec:.4f}, "
                f"p99={stats.p99_batch_sec:.4f}"
            )

        if stats_by_split:
            total_processed = sum(
                split_stats.processed_samples for split_stats in stats_by_split.values()
            )
            total_elapsed = sum(
                split_stats.elapsed_sec for split_stats in stats_by_split.values()
            )
            total_errors = sum(
                split_stats.error_samples for split_stats in stats_by_split.values()
            )
            aggregate_sps = (
                total_processed / total_elapsed if total_elapsed > 0 else 0.0
            )
            aggregate_error_rate = (
                total_errors / total_processed if total_processed > 0 else 0.0
            )
            print(
                "\nAggregate: "
                f"{total_processed} samples in {total_elapsed:.1f}s "
                f"({aggregate_sps:.2f} samples/s), "
                f"errors={total_errors} ({100.0 * aggregate_error_rate:.2f}%)"
            )

    print("\nUpdating parquet files...")
    total_removed = 0
    for parquet_path, bad_paths in bad_paths_by_parquet.items():
        removed = clean_parquet_file(
            parquet_path=parquet_path,
            bad_paths=bad_paths,
            dry_run=args.dry_run,
        )
        total_removed += removed
        action = "Would remove" if args.dry_run else "Removed"
        print(f"- {action} {removed} rows from {parquet_path}")

    if args.dry_run:
        print(f"\nDry run complete. Rows that would be removed: {total_removed}")
    else:
        print(f"\nDone. Total rows removed: {total_removed}")


if __name__ == "__main__":
    main()
