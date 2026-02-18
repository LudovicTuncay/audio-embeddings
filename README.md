# Audio Embeddings with Lightning & Hydra

This project is a clean, modular, and scalable implementation of audio embedding models using **PyTorch Lightning** and **Hydra**. It is designed to be easily extensible and runnable on local or cluster environments. It is based on the [Audio-JEPA](https://github.com/LudovicTuncay/Audio-JEPA) implementation and therefore implements the Audio-JEPA architecture. Other architecture can and will be added in the future.

## 🎯 Goal

The goal of this project is to provide a robust codebase for training and experimenting with audio embedding models. Key features include:
- **Modular Architecture**: Components like Spectrogram, Masking, and ViT are decoupled.
- **Configurable Positional Embeddings**: Support for **RoPE** (2D Rotary Embeddings), **SinCos** (2D Sinusoidal), and **Learnable** embeddings.
- **Hydra Configuration**: flexible experiment management via hierarchical config files.
- **Lightning Trainer**: Simplified training loop, logging, and checkpointing.
- **Modern Tooling**: Uses `uv` for fast and reliable dependency management.

## 🚀 Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

1.  **Install `uv`** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd audio-embeddings
    ```

3.  **Install dependencies**:
    ```bash
    uv sync
    ```
    For development/testing tools (including `pytest`), sync all groups:
    ```bash
    uv sync --all-groups
    ```

4.  **Enable shared git hooks** (runs `uv sync` after merge/checkout/rewrite):
    ```bash
    git config core.hooksPath .githooks
    ```

## 🏃 Usage

### Basic Training
To start training with the default configuration:
```bash
uv run src/train.py
```

### Common Commands
Run on GPU with Weights & Biases logging:
```bash
uv run src/train.py trainer=gpu logger=wandb
```

Override hyperparameters on the command line:
```bash
uv run src/train.py data.batch_size=64 trainer.max_epochs=50
```

### UPS WebDataset (Local Shards)
Train directly from local UPS tar shards on a cluster filesystem:
```bash
uv run src/train.py \
    data=ups_webdataset \
    trainer=cpu \
    +trainer.fast_dev_run=True \
    data.shard_globs='[/mnt/ups/audio/*.tar,/mnt/ups/audio2/*.tar]'
```
The loader expects UPS shard samples under the `mp3` key.

### Configurable Positional Embeddings
You can switch between different positional embedding strategies easily:

**RoPE**:
```bash
uv run src/train.py model.net.encoder.pos_embed_type=rope
```

### Offline WandB Logging with Model Checkpoints
To run training offline but still have model checkpoints staged for upload (which standard WandB restricts):

```bash
uv run src/train.py \
    logger=wandb \
    logger.wandb.offline=True \
    logger.wandb.log_model=False \
    +callbacks.wandb_offline_checkpoint._target_=src.callbacks.wandb_callbacks.WandbOfflineCheckpointCallback \
    trainer=gpu trainer.devices=1 \
    data.batch_size=128 trainer.max_epochs=100
```
These checkpoints will be uploaded when you run `wandb sync`.


**2D SinCos**:
```bash
uv run src/train.py ++model.net.encoder.pos_embed_type=sincos ++model.net.predictor.pos_embed_type=sincos
```

**Learnable**:
```bash
uv run src/train.py ++model.net.encoder.pos_embed_type=learnable ++model.net.predictor.pos_embed_type=learnable
```

## 📂 Project Structure

```text
├── configs/                 # Hydra configuration files
│   ├── callbacks/           # Callback configs (checkpoints, early stopping)
│   ├── data/                # Data configs (AudioSet, etc.)
│   ├── logger/              # Logger configs (WandB, Tensorboard)
│   ├── model/               # Model configs (AudioJEPA parameters)
│   ├── trainer/             # Trainer configs (CPU, GPU, strategies)
│   └── train.yaml           # Main configuration entry point
├── src/
│   ├── data/                # Data loading logic
│   │   └── audioset_datamodule.py  # AudioSet DataModule & Dataset
│   ├── models/              # Model architectures
│   │   ├── components/      # Reusable blocks
│   │   │   ├── masking.py   # Masking generators
│   │   │   ├── patch_embed.py # Patchification
│   │   │   ├── rope.py      # 2D Rotary Embeddings
│   │   │   ├── spectrogram.py # Audio preprocessing
│   │   │   └── vit.py       # Vision Transformer (Student/Teacher/Predictor)
│   │   └── audio_jepa_module.py # Main LightningModule
│   ├── utils/               # Utility functions
│   └── train.py             # Training entry point
├── scripts/                 # Helper scripts
├── tests/                   # Verification tests
├── pyproject.toml           # Project dependencies
└── README.md                # This file
```

## 🛠️ Extensibility

### Adding a New Model
1.  Create your model components in `src/models/components/`.
2.  Create a new LightningModule in `src/models/` (or update `AudioJEPAModule`).
3.  Create a new config file in `configs/model/my_new_model.yaml`.
4.  Run with `uv run src/train.py model=my_new_model`.

### Adding a New Dataset
1.  Create a new DataModule in `src/data/`.
2.  Create a new config file in `configs/data/my_dataset.yaml`.
3.  Run with `uv run src/train.py data=my_dataset`.

### Adding Functionalities
-   **Callbacks**: Add custom callbacks in `src/callbacks/` (if needed) or use existing Lightning callbacks, and configure them in `configs/callbacks/`.
-   **Metrics**: Add metrics logging in `training_step` or `validation_step` inside `src/models/audio_jepa_module.py`.

## 🧪 Testing
The project uses a two-tier testing workflow:

1. Fast `pytest` checks by default.
2. Heavier `integration`/`data` checks on demand.

`pytest` is defined in the `dev` dependency group in `pyproject.toml`, so examples below use `--group dev`.

Run the default fast pytest suite:
```bash
uv run --group dev pytest
```

Run a single pytest file:
```bash
uv run --group dev pytest tests/test_audio_utils.py -q
```

Run slower integration/data checks:
```bash
uv run --group dev pytest -m "integration or data"
```

Keep script-based verifications for manual/component checks:
```bash
uv run tests/verify_rope.py
uv run tests/verify_custom_rope.py
uv run tests/verify_data.py
```

## 📜 License

This repository is licensed under the MIT License. See `LICENSE`.

This repository also includes vendored third-party code under Apache-2.0.
See `THIRD_PARTY_LICENSES.md` and `licenses/APACHE-2.0-LIGHTNING.txt` for details.
