"""Shared pytest fixtures for Hydra config composition."""

from pathlib import Path

import pytest
import rootutils
from hydra import compose
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from omegaconf import open_dict


@pytest.fixture(scope="function")
def cfg_train(tmp_path: Path) -> DictConfig:
    """Compose a train config that is safe for local tests."""
    GlobalHydra.instance().clear()

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])

    with open_dict(cfg):
        cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1
        cfg.trainer.logger = False
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.extras.print_config = False
        cfg.extras.enforce_tags = False
        cfg.logger = None

    return cfg


@pytest.fixture(autouse=True)
def clear_hydra_global_state() -> None:
    """Ensure tests remain isolated when composing Hydra configs."""
    yield
    GlobalHydra.instance().clear()
