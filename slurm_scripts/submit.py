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
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={gpus}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log

set -x

export MPLBACKEND=Agg

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

echo "Starting job {job_name} on $(hostname)"
echo "Experiment: {experiment}"

srun .venv/bin/python -u -O src/train.py \\
    experiment={experiment} \\
    trainer.devices={gpus} \\
    trainer.strategy={strategy} \\
    trainer.max_time="{max_time}" \\
    logger.wandb.name="{wandb_name}" \\
    {extra_args}
"""

def parse_slurm_time(time_str):
    """Parses Slurm time string into a timedelta object.
    Formats: "MM", "MM:SS", "HH:MM:SS", "D-HH", "D-HH:MM", "D-HH:MM:SS"
    """
    days = 0
    if '-' in time_str:
        days_str, time_str = time_str.split('-')
        days = int(days_str)

    parts = list(map(int, time_str.split(':')))
    
    if len(parts) == 1: # MM
        minutes = parts[0]
        hours = 0
        seconds = 0
    elif len(parts) == 2: # MM:SS
        minutes, seconds = parts
        hours = 0
    elif len(parts) == 3: # HH:MM:SS
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

def format_steps(steps_str):
    if not steps_str or not steps_str.isdigit():
        return steps_str
    
    steps = int(steps_str)
    if steps >= 1000000:
        return f"{steps//1000000}m"
    if steps >= 1000:
        return f"{steps//1000}k"
    return str(steps)

def generate_wandb_name(config_path, num_gpus, suffix=None):
    try:
        with open(config_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}. Cannot auto-generate name.")
        return "experiment"

    # Extract values using regex
    model = parse_config_value(content, r"override /model:\s*(\S+)")
    dataset = parse_config_value(content, r"override /data:\s*(\S+)")
    batch_size = parse_config_value(content, r"batch_size:\s*(\d+)")
    max_steps = parse_config_value(content, r"max_steps:\s*(\d+)")
    
    # Construct name parts
    parts = []
    
    if model: parts.append(model)
    if dataset: parts.append(dataset)
    
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
    parser = argparse.ArgumentParser(description="Generate and submit Slurm jobs for Audio Embeddings.")
    parser.add_argument("experiment", type=str, help="Experiment config path (e.g., audio_jepa/baseline)")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to request (default: 1)")
    parser.add_argument("--time", type=str, default="20:00:00", help="Time limit (HH:MM:SS) (default: 20:00:00)")
    parser.add_argument("--suffix", type=str, help="Optional suffix for WandB run name")
    parser.add_argument("--dry-run", action="store_true", help="Print the generated script without submitting")
    
    args, unknown = parser.parse_known_args()
    
    # 1. Configuration Logic
    if args.gpus > 1:
        strategy = "ddp"
        # Sync BatchNorm is usually recommended for DDP
        extra_args_list = ["trainer.sync_batchnorm=True"]
    else:
        strategy = "auto"
        extra_args_list = []
        
    # Append any unknown arguments passed to the script (e.g. model.rq_lambda=0.5)
    if unknown:
        extra_args_list.extend(unknown)
        
    extra_args = " ".join(extra_args_list)
        
    # Get absolute path of current working directory
        
    # Get absolute path of current working directory
    workdir = os.path.abspath(os.getcwd())

    # 2. Generate WandB Name
    # Assume config is in configs/experiment/{experiment}.yaml
    config_path = os.path.join(workdir, "configs", "experiment", f"{args.experiment}.yaml")
    wandb_name = generate_wandb_name(config_path, args.gpus, args.suffix)
    
    # Use WandB name as Job Name (consistent naming)
    job_name = wandb_name
    
    # 3. Calculate Trainer Max Time (Time - 10 minutes)
    try:
        slurm_time_td = parse_slurm_time(args.time)
        buffer_time = timedelta(minutes=10)
        
        # Ensure we don't go negative
        if slurm_time_td > buffer_time:
            max_time_td = slurm_time_td - buffer_time
        else:
            print(f"Warning: Requested time {args.time} is less than buffer (10m). Using full time.")
            max_time_td = slurm_time_td
            
        max_time_str = format_timedelta(max_time_td)
    except Exception as e:
        print(f"Warning: Could not parse time '{args.time}'. Using original string for max_time. Error: {e}")
        max_time_str = args.time

    # 4. Fill Template
    script_content = TEMPLATE.format(
        job_name=job_name,
        time=args.time,
        gpus=args.gpus,
        workdir=workdir,
        experiment=args.experiment,
        strategy=strategy,
        wandb_name=wandb_name,
        max_time=max_time_str,
        extra_args=extra_args
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
        result = subprocess.run(["sbatch", filename], check=True, capture_output=True, text=True)
        print(f"Submission successful: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print("Error: Submission failed!")
        print(f"Stderr: {e.stderr}")
        # Optionally delete the failed script? Keeping it for debug is usually better.

if __name__ == "__main__":
    main()
