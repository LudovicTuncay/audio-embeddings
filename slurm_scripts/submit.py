#!/usr/bin/env python3
import argparse
import os
import subprocess
import re
from datetime import datetime, timedelta

# Slurm Script Template
# Adapt directives based on your cluster configuration
TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=ojz@h100
#SBATCH --constraint=h100
#SBATCH --qos={qos}
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={gpus}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log

set -euxo pipefail

export MPLBACKEND=Agg

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh || true
fi

module load {ffmpeg_module}

FFMPEG_BIN=$(command -v ffmpeg || true)
if [ -n "$FFMPEG_BIN" ]; then
    FFMPEG_ROOT=$(dirname "$(dirname "$FFMPEG_BIN")")
    export LD_LIBRARY_PATH="${FFMPEG_ROOT}/lib:${{LD_LIBRARY_PATH}}"
fi

if [ -n "${EBROOTFFMPEG:-}" ]; then
    export LD_LIBRARY_PATH="${EBROOTFFMPEG}/lib:${{LD_LIBRARY_PATH}}"
fi

cd {workdir}

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TMPDIR=$SCRATCH
export TEMP=$SCRATCH
export TMP=$SCRATCH
export PROJECT_ROOT={workdir}

# Ensure log directory exists
mkdir -p logs/slurm

source .venv/bin/activate

# Configuration Info
# Experiment: {experiment}
# GPUs: {gpus}
# Strategy: {strategy}
# WandB Name: {wandb_name}
# Trainer Max Time: {max_time}
# FFmpeg module: {ffmpeg_module}

echo "Starting job {job_name} on $(hostname)"
echo "Experiment: {experiment}"
echo "FFmpeg binary: $(command -v ffmpeg || echo 'not found')"
ffmpeg -version | head -n 1

srun .venv/bin/python -u -O src/train.py \\
    experiment={experiment} \\
    trainer.devices={gpus} \\
    +trainer.strategy={strategy} \\
    +trainer.max_time="{max_time}" \\
    ++logger.wandb.name="{wandb_name}" \\
    {extra_args}
"""


def parse_slurm_time(time_str):
    """Parses Slurm time string into a timedelta object.
    Formats: "MM", "MM:SS", "HH:MM:SS", "D-HH", "D-HH:MM", "D-HH:MM:SS"
    """
    days = 0
    if "-" in time_str:
        days_str, time_str = time_str.split("-")
        days = int(days_str)

    parts = list(map(int, time_str.split(":")))

    if len(parts) == 1:  # MM
        minutes = parts[0]
        hours = 0
        seconds = 0
    elif len(parts) == 2:  # MM:SS
        minutes, seconds = parts
        hours = 0
    elif len(parts) == 3:  # HH:MM:SS
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Invalid time format: {time_str}")

    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def format_timedelta(td):
    """Formats timedelta back to DD:HH:MM:SS string for Lightning"""
    total_seconds = int(td.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days:02}:{hours:02}:{minutes:02}:{seconds:02}"


def parse_config_value(content, pattern):
    match = re.search(pattern, content)
    return match.group(1).strip() if match else None


def select_qos(time_limit: timedelta) -> str:
    two_hours = timedelta(hours=2)
    twenty_hours = timedelta(hours=20)
    one_hundred_hours = timedelta(hours=100)

    if time_limit <= two_hours:
        return "qos_gpu_h100-dev"
    if time_limit <= twenty_hours:
        return "qos_gpu_h100-t3"
    if time_limit <= one_hundred_hours:
        return "qos_gpu_h100-t4"

    raise ValueError(
        "Requested time exceeds maximum supported QoS window (100h). "
        "Please request 100:00:00 or less."
    )


def format_steps(steps_str):
    if not steps_str or not steps_str.isdigit():
        return steps_str

    steps = int(steps_str)
    if steps >= 1000000:
        return f"{steps // 1000000}m"
    if steps >= 1000:
        return f"{steps // 1000}k"
    return str(steps)


def generate_wandb_name(config_path, num_gpus, suffix=None):
    try:
        with open(config_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(
            f"Warning: Config file not found at {config_path}. Cannot auto-generate name."
        )
        return "experiment"

    # Extract values using regex
    model = parse_config_value(content, r"override /model:\s*(\S+)")
    dataset = parse_config_value(content, r"override /data:\s*(\S+)")
    batch_size = parse_config_value(content, r"batch_size:\s*(\d+)")
    max_steps = parse_config_value(content, r"max_steps:\s*(\d+)")

    # Construct name parts
    parts = []

    if model:
        parts.append(model)
    if dataset:
        parts.append(dataset)

    if max_steps:
        parts.append(format_steps(max_steps))

    if batch_size:
        parts.append(f"{batch_size}x{num_gpus}bs")

    if suffix:
        parts.append(suffix)

    # Fallback if parsing failed completely
    if not parts:
        return "experiment"

    return "-".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate and submit Slurm jobs for Audio Embeddings. "
            "WandB run names are generated from the experiment config "
            "(model, data, max_steps, batch_size x GPUs), plus optional suffix."
        )
    )
    parser.add_argument(
        "experiment",
        type=str,
        help="Experiment config path (e.g., audio_jepa/baseline)",
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs to request (default: 1)"
    )
    parser.add_argument(
        "--time",
        type=str,
        default="20:00:00",
        help="Time limit (HH:MM:SS) (default: 20:00:00)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        help=(
            "Optional suffix for WandB run name. "
            "Base name is derived from the experiment config: "
            "model + data + max_steps (k/m) + batch_size x GPUs."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated script without submitting",
    )
    parser.add_argument(
        "--ffmpeg-module",
        type=str,
        default="ffmpeg/7.1-cuda",
        help=(
            "FFmpeg environment module to load in the job script "
            "(default: ffmpeg/7.1-cuda)."
        ),
    )

    args, unknown = parser.parse_known_args()

    # 1. Configuration Logic
    if args.gpus > 1:
        strategy = "ddp"
        # Sync BatchNorm is usually recommended for DDP
        # Using +trainer.sync_batchnorm to ensure we append it even if it doesn't exist
        extra_args_list = ["+trainer.sync_batchnorm=True"]
    else:
        strategy = "auto"
        extra_args_list = []

    # Append any unknown arguments passed to the script (e.g. model.rq_lambda=0.5)
    if unknown:
        for arg in unknown:
            if arg.startswith(("+", "~")):
                extra_args_list.append(arg)
            elif "=" in arg:
                # If it's an assignment, use ++ to Force Add/Override
                # This prevents "ConfigAttributeError" if the key isn't in the struct
                # and works fine if it IS in the struct.
                extra_args_list.append("++" + arg)
            else:
                extra_args_list.append(arg)

    extra_args = " ".join(extra_args_list)

    # Get absolute path of current working directory

    # Get absolute path of current working directory
    workdir = os.path.abspath(os.getcwd())

    # 2. Generate WandB Name
    # Assume config is in configs/experiment/{experiment}.yaml
    config_path = os.path.join(
        workdir, "configs", "experiment", f"{args.experiment}.yaml"
    )
    wandb_name = generate_wandb_name(config_path, args.gpus, args.suffix)

    # Use WandB name as Job Name (consistent naming)
    job_name = wandb_name

    # 3. Select QoS and calculate Trainer Max Time (Time - 10 minutes)
    try:
        slurm_time_td = parse_slurm_time(args.time)
        qos = select_qos(slurm_time_td)
        buffer_time = timedelta(minutes=10)

        # Ensure we don't go negative
        if slurm_time_td > buffer_time:
            max_time_td = slurm_time_td - buffer_time
        else:
            print(
                f"Warning: Requested time {args.time} is less than buffer (10m). Using full time."
            )
            max_time_td = slurm_time_td

        max_time_str = format_timedelta(max_time_td)
    except Exception as e:
        raise ValueError(f"Invalid --time value '{args.time}': {e}") from e

    # 4. Fill Template
    script_content = TEMPLATE.format(
        job_name=job_name,
        qos=qos,
        time=args.time,
        gpus=args.gpus,
        workdir=workdir,
        experiment=args.experiment,
        strategy=strategy,
        wandb_name=wandb_name,
        max_time=max_time_str,
        ffmpeg_module=args.ffmpeg_module,
        extra_args=extra_args,
    )

    # 5. Handle Dry Run
    if args.dry_run:
        print("--- Dry Run: Generated Slurm Script ---")
        print(script_content)
        print("---------------------------------------")
        return

    # 4. Write to Temporary File
    # Create a hidden temp directory for scripts if it doesn't exist
    script_dir = os.path.join(workdir, "slurm_scripts", ".generated")
    os.makedirs(script_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(script_dir, f"submit_{job_name}_{timestamp}.slurm")

    with open(filename, "w") as f:
        f.write(script_content)

    print(f"Generated script: {filename}")

    # 5. Submit to Slurm
    try:
        # Submit the script
        result = subprocess.run(
            ["sbatch", filename], check=True, capture_output=True, text=True
        )
        print(f"Submission successful: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print("Error: Submission failed!")
        print(f"Stderr: {e.stderr}")
        # Optionally delete the failed script? Keeping it for debug is usually better.


if __name__ == "__main__":
    main()
