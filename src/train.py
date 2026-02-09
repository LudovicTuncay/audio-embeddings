import rootutils
import hydra
from omegaconf import DictConfig
import lightning as L
import torch
from pathlib import Path
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import List, Dict, Any

# Setup root
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import instantiate_callbacks, instantiate_loggers, RankedLogger, extras  # noqa: E402

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

    callbacks_cfg = cfg.get("callbacks")
    if (
        isinstance(callbacks_cfg, DictConfig)
        and "model_checkpoint" in callbacks_cfg
        and callbacks_cfg.model_checkpoint is None
    ):
        log.warning(
            "`callbacks.model_checkpoint` is null in the composed config. "
            "Lightning will use its default ModelCheckpoint callback, which may not "
            "save `last.ckpt` and can change filename conventions. Remove the null "
            "override or set explicit checkpoint fields in the experiment config."
        )

    if cfg.get("train") and not any(
        isinstance(callback, ModelCheckpoint) for callback in callbacks
    ):
        log.warning(
            "No explicit ModelCheckpoint callback was instantiated from config; "
            "Lightning default checkpointing behavior will be used."
        )

    log.info("Instantiating loggers...")
    logger: List[L.Logger] = instantiate_loggers(cfg.get("logger"))

    # Set float32 matmul precision for Tensor Cores
    torch.set_float32_matmul_precision("medium")

    # Log config tree and .hydra folder to wandb
    for lg in logger:
        if isinstance(lg, WandbLogger):
            # check if config_tree.log exists
            config_tree_path = Path(cfg.paths.output_dir, "config_tree.log")
            if config_tree_path.exists():
                log.info("Logging config tree to WandB...")
                lg.experiment.save(
                    str(config_tree_path), policy="now", base_path=cfg.paths.output_dir
                )

            # Upload .hydra folder contents
            hydra_dir = Path(cfg.paths.output_dir, ".hydra")
            if hydra_dir.exists() and hydra_dir.is_dir():
                log.info("Logging .hydra folder to WandB...")
                for hydra_file in hydra_dir.iterdir():
                    if hydra_file.is_file():
                        lg.experiment.save(
                            str(hydra_file),
                            policy="now",
                            base_path=cfg.paths.output_dir,
                        )

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
