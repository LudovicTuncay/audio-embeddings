import argparse
from collections import defaultdict
from collections.abc import Iterable

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


def scan_split_for_failures(
    split_name: str,
    dataset: YT1BDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> set[str]:
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
    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task(f"Scanning {split_name}", total=len(dataset))

        for batch in dataloader:
            for sample in batch:
                if sample.get("error", False):
                    sample_index = int(sample["index"])
                    bad_paths.add(dataset.paths[sample_index])
                progress.advance(task_id, 1)

    return bad_paths


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
        default="/lustre/fsmisc/dataset/YT-Temporal-1B",
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

    for split_name, dataset, parquet_path in split_specs:
        if dataset is None:
            print(f"Skipping {split_name}: parquet not found at {parquet_path}")
            continue

        bad_paths = scan_split_for_failures(
            split_name=split_name,
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        bad_counts_by_split[split_name] = len(bad_paths)
        bad_paths_by_parquet[parquet_path].update(bad_paths)

    print("\nFailure counts by split:")
    for split_name in ["train", "val", "test"]:
        if split_name in bad_counts_by_split:
            print(f"- {split_name}: {bad_counts_by_split[split_name]}")

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
