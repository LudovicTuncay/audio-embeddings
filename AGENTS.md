# AGENTS Guide - audio-embeddings

This file is for coding agents working in this repository.
Follow these repo-specific rules over generic defaults.

## 1) Environment Snapshot
- Python: `>=3.12` (from `pyproject.toml`).
- Dependency manager: `uv`.
- Main stack: PyTorch, PyTorch Lightning, Hydra, OmegaConf.
- Project root marker: `.project-root`.
- Main entrypoint: `src/train.py`.

## 2) Cursor / Copilot Rule Files
- Checked `.cursor/rules/`: not present.
- Checked `.cursorrules`: not present.
- Checked `.github/copilot-instructions.md`: not present.
- Therefore, no additional Cursor/Copilot rule files are currently enforced.

## 3) Install / Setup Commands
```bash
uv sync
uv sync --all-groups
uv run <command>
uv add <package>
```

## 4) Build / Train / Eval Commands
There is no separate "build" step (this is a training codebase).
Use quick-run training as the integration sanity check.
```bash
uv run src/train.py
uv run src/train.py trainer.fast_dev_run=True
uv run src/train.py trainer=cpu trainer.fast_dev_run=True
uv run src/train.py experiment=local/audio_jepa
uv run src/train.py trainer.max_epochs=10 data.batch_size=32 model.optimizer.lr=1e-4
```
Cluster-style execution (existing project pattern):
```bash
srun .venv/bin/python -u -O src/train.py experiment=cluster_jepa_audioset_rope +trainer.max_time="00:19:50:00"
```

## 5) Lint / Formatting / Static Checks
Use the commands below as pragmatic checks:
```bash
uv run pre-commit run --all-files
uv run pre-commit run ruff-check --all-files
uv run pre-commit run ruff-format --all-files
uv run python -m compileall src
```
Ruff is configured via `.pre-commit-config.yaml` and runs both lint fixes and formatting.

## 6) Test Commands (Including Single Test)
Primary validation is now pytest-based with marker tiers:
- Default run excludes `slow`, `data`, and `gpu` tests (configured in `pyproject.toml`).
- Use `integration`/`data` markers for heavier end-to-end checks.

Run default fast tests:
```bash
uv run --group dev pytest
```

Run a single pytest file or test:
```bash
uv run --group dev pytest tests/test_audio_utils.py -q
uv run --group dev pytest tests/test_train.py::test_train_fast_dev_run_cpu -q
```

Run slower integration/data checks:
```bash
uv run --group dev pytest -m "integration or data"
```

Script-based verifications remain useful for targeted manual checks:
```bash
uv run tests/verify_rope.py
uv run tests/verify_custom_rope.py
uv run tests/verify_data.py
```
Additional quick sanity commands:
```bash
uv run src/train.py trainer.fast_dev_run=True
uv run src/train.py trainer=cpu trainer.fast_dev_run=True
uv run --group dev pytest tests/test_model_forward.py -q
uv run --group dev pytest tests/test_lr_scheduler.py -q
```
Notes:
- Use `uv run --group dev pytest ...` for pytest commands.
- Keep `tests/verify_*.py` for lightweight debug/inspection workflows.

## 7) Repository Architecture Expectations
- `configs/`: Hydra composition (trainer/data/model/logger/callbacks/experiment).
- `src/train.py`: orchestration only (instantiate and run).
- `src/models/`: LightningModules (high-level training logic).
- `src/models/components/`: reusable `nn.Module` building blocks.
- `src/data/`: DataModules/Datasets and collate logic.
- `src/utils/`: logging, instantiation, wrappers, scheduler helpers.
When possible, prefer config changes over hardcoded Python changes.

## 8) Code Style Guidelines
### Imports
- Group imports as: standard library -> third-party -> local `src.*`.
- Keep one import per line unless importing multiple names from same module.
- Avoid wildcard imports.
- Prefer absolute imports from `src...`.

### Formatting
- Use 4-space indentation and readable line lengths.
- Keep functions small; extract helpers for complex logic.
- Do not introduce unrelated reformatting in touched files.
- Keep comments for non-obvious intent, not obvious mechanics.

### Typing
- Type hints are expected for function arguments and return values.
- Use concrete tensor/container types when practical.
- Use `Optional[T]` / `T | None` consistently within a file.
- For dict-like configs, type as `DictConfig` when passing Hydra config objects.

### Naming
- `snake_case`: functions, variables, module filenames.
- `PascalCase`: classes (`AudioJEPAModule`, `AudioSetDataModule`).
- `UPPER_SNAKE_CASE`: constants.
- Prefer descriptive names (`mask_indices`) over short names (`m2`) except local math temporaries.

### PyTorch / Lightning / Hydra Conventions
- Keep heavy compute out of `__init__` where possible.
- `forward()` for inference logic; training behavior in `training_step()`.
- Use `self.log(...)` with explicit flags (`on_step`, `on_epoch`, `prog_bar`, `batch_size`).
- Instantiate components through Hydra (`hydra.utils.instantiate`).
- Expose tunable parameters in config files, not hardcoded literals.

### Error Handling and Validation
- Raise informative `ValueError` / `RuntimeError` for invalid config/state.
- Validate critical tensor assumptions with assertions or explicit checks.
- Prefer logger/warnings over bare `print()` in new code.
- For file I/O, prefer `pathlib.Path` and existence checks.

### Data and Paths
- Do not hardcode absolute machine paths.
- Use `rootutils.setup_root(..., indicator=".project-root", pythonpath=True)` in entrypoints/scripts when needed.
- Respect `cfg.paths.*` outputs for logs/checkpoints/artifacts.

## 9) Agent Workflow Rules
- Reuse existing components before adding new abstractions.
- Keep `src/train.py` generic; place model/data logic in dedicated modules.
- Prefer minimal, focused diffs.
- Update configs and docs when behavior changes.
- Validate with the smallest meaningful command first (`fast_dev_run`, single test), then broader checks.

## 10) Git / Change Hygiene
- Do not revert unrelated local changes.
- Keep commits scoped to one concern.
- Write clear commit messages describing intent.
- Prefer Conventional Commit-like format: `type(scope): intent`.
- Common types in this repo: `feat`, `fix`, `conf`, `build`, `docs`, `style`, `chore`.
- Never commit secrets, credentials, or environment-specific absolute paths.

## 11) Practical Agent Defaults
- Prefer reusing existing modules over creating new abstractions.
- Keep edits local to the requested change; avoid drive-by refactors.
- Run the smallest useful verification command after changes.
- If you touch training logic, run at least one fast training sanity check.
- If you touch model components, run relevant verify script(s) in `tests/`.
- If you touch Hydra config wiring, run a config-backed entry command via `uv run src/train.py ...`.

## 12) Common Pitfalls
- Avoid hardcoding data paths; use config (`cfg.paths`, data config fields).
- Avoid printing in new code paths; use ranked loggers/warnings.
- Avoid putting heavy tensor compute in constructors.
- Avoid bypassing Hydra by manually instantiating configurable components.
- Avoid changing unrelated formatting in files you touch.
