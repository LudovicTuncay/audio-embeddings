#!/bin/bash
#SBATCH --account=ojz@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --output=logs/slurm/AudioJEPA-%j.out
#SBATCH --error=logs/slurm/AudioJEPA-%j.err

# --------- PROGRAM START ------------
set -x

cd ${SLURM_SUBMIT_DIR}

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TMPDIR=$SCRATCH
export TEMP=$SCRATCH
export TMP=$SCRATCH
export COLUMNS=200
export PROJECT_ROOT=${SLURM_SUBMIT_DIR}

# Ensure log directory exists
mkdir -p logs/slurm

source .venv/bin/activate

# Execute training
# Using audioset_baseline experiment but overriding for distributed training on 4 GPUs
srun .venv/bin/python -u -O src/train.py \
    experiment=jepa_audioset_baseline \
    trainer=ddp trainer.devices=4 trainer.max_time="00:19:55:00" \
    data.batch_size=128
