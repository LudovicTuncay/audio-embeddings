# Agent Guidelines for Audio Embeddings Repository

This repository implements audio embedding models using **PyTorch Lightning** and **Hydra**.
Follow these guidelines to ensure consistency and maintainability.

## 1. Environment & Build

- **Dependency Management**: uses `uv` (modern Python package manager).
  - Install dependencies: `uv sync` (creates `.venv`).
  - Add dependency: `uv add <package>`.
  - Activate environment: `source .venv/bin/activate`.
- **Python Version**: `>=3.12`.
- **Hardware**: Code is optimized for GPU training but supports CPU for debugging (`accelerator='cpu'`).

## 2. Running Commands

### Training
The entry point is `src/train.py`. It uses Hydra for configuration composition.

#### Local Training
For local development, use `uv run`:
```bash
# Default training
uv run src/train.py

# Train with specific experiment config
uv run src/train.py experiment=example

# Debug run (fast, no logging, 1 epoch)
uv run src/train.py trainer.fast_dev_run=True

# Overriding hyperparameters via CLI
uv run src/train.py trainer.max_epochs=10 data.batch_size=32 model.optimizer.lr=1e-4
```

#### Cluster Training (Jean Zay)
For experiments on the Jean Zay cluster, use `srun` with the `.venv` python:
```bash
# Example cluster job submission
srun .venv/bin/python -u -O src/train.py experiment=cluster_jepa_audioset_rope +trainer.max_time="00:19:50:00"
```

### Testing & Verification
The primary way to verify code changes is by running the standalone verification scripts located in `tests/`. These scripts check specific components (like RoPE embeddings, DataModules, etc.) without requiring a full test harness.

```bash
# Verify RoPE implementation
uv run tests/verify_rope.py
uv run tests/verify_custom_rope.py

# Verify DataModule
uv run tests/verify_data.py
```

Note: There are `pytest` style tests in `tests/` (e.g., `test_train.py`), but they are not currently part of the default CI/dev loop. To run them, you would need to install pytest manually (`uv add --dev pytest`).

## 3. Code Style & Conventions

### General Python
- **Formatting**: Code should be clean and readable. Use standard Python formatting (4 spaces indentation).
- **Type Hints**: **Mandatory**. All function arguments and return values must be typed.
  ```python
  def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
      ...
  ```
- **Imports**: Group imports: standard library, third-party (torch, lightning, hydra), then local (`src.*`).
  ```python
  import os
  from typing import Dict
  
  import torch
  import lightning as L
  
  from src.models.components.vit import ViT
  ```

### Project Structure
- **Configs** (`configs/`):
  - `train.yaml`: Main entry config.
  - `model/`: Model architecture configs.
  - `data/`: DataModule configs.
  - `trainer/`: Lightning Trainer configs.
- **Source** (`src/`):
  - `src/models/`: LightningModules (e.g., `AudioJEPAModule`). Keep these high-level.
  - `src/models/components/`: Pure PyTorch `nn.Modules`. Implement specific layers/blocks here.
  - `src/data/`: LightningDataModules. Handle downloading, parsing, and `DataLoader` creation.
  - `src/utils/`: Utilities for logging, instantiation, etc.
- **Paths**: NEVER hardcode paths. Use `rootutils` to locate project root dynamically.
  ```python
  import rootutils
  root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
  ```

### PyTorch Lightning & Hydra Best Practices
- **Instantiation**: Use `hydra.utils.instantiate(config)` to create objects from config. This allows swapping implementations via config without changing code.
- **LightningModule**:
  - `__init__`: Accept configuration/args. Do not perform heavy computation here.
  - `forward`: Inference logic only.
  - `training_step`: Loss calculation and logging.
  - `configure_optimizers`: Define optimizers and schedulers.
- **Logging**:
  - Use `self.log("name", value, ...)` within LightningModules.
  - Prefer explicit logging arguments: `on_step=True`, `on_epoch=True`, `prog_bar=True`.
- **Config**: Do not hardcode hyperparameters in Python. Expose them in `__init__` and set via Hydra config files.

### Error Handling & Safety
- **File I/O**: Use `pathlib.Path`. Check for existence before reading.
- **Assertions**: Use assertions to validate shapes and tensor types in complex modules (e.g., `assert x.shape[-1] == self.embed_dim`).
- **Exceptions**: Raise informative `ValueError` or `RuntimeError` if configuration is invalid.

## 4. Development Workflow

1.  **Analyze**: Before writing code, search for existing components to reuse (`src/models/components`).
2.  **Config First**: Check if the desired change can be achieved via `configs/` modifications.
3.  **Verify**:
    - Run `python src/train.py trainer.fast_dev_run=True` to verify the training loop works.
    - Run `pytest` to ensure no regressions.
4.  **Lint**: Ensure code follows the existing style.

## 5. Specific Repository Rules
- **Do not hardcode paths**. Use `pathlib` and `rootutils` or Hydra `cfg.paths`.
- **Do not introduce global state**. Avoid global variables.
- **Keep `src/train.py` generic**. It should just glue configuration to execution. logic should be in Modules or Callbacks.
- **Hydra Composition**: Leverage Hydra's composition. Do not manually parse args.

## 6. Troubleshooting
- **Hydra Overrides**: If parameters aren't changing, check if they are correctly nested in the config structure.
- **Shape Errors**: Use `print(x.shape)` or `import pdb; pdb.set_trace()` inside `training_step` to debug tensor mismatches.
- **DDP Issues**: When using multiple GPUs (`trainer.devices > 1`), ensure logic in `LightningModule` is rank-safe (use `rank_zero_only` for logging/saving).

## 7. Git & Commit
- **Commits**: Write clear, descriptive commit messages.
- **Scope**: Keep commits focused on a single change (refactor, feature, or fix).
