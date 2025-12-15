import rootutils
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
import torch
from pathlib import Path
from lightning.pytorch.loggers import WandbLogger
from typing import List, Dict, Any

# Setup root
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import instantiate_callbacks, instantiate_loggers, RankedLogger, extras

log = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Dict[str, Any]:
    # Set seed
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Applies optional utilities
    extras(cfg)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[L.Logger] = instantiate_loggers(cfg.get("logger"))

    # Set float32 matmul precision for Tensor Cores
    torch.set_float32_matmul_precision("medium")

    # Log config tree to wandb
    for lg in logger:
        if isinstance(lg, WandbLogger):
            # check if config_tree.log exists
            config_tree_path = Path(cfg.paths.output_dir, "config_tree.log")
            if config_tree_path.exists():
                log.info("Logging config tree to WandB...")
                lg.experiment.save(str(config_tree_path), policy="now", base_path=cfg.paths.output_dir)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    return object_dict

if __name__ == "__main__":
    main()
