import os
import glob
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only

class WandbOfflineCheckpointCallback(Callback):
    """
    Custom callback to log model checkpoints to WandB even when offline=True.
    Lightning's WandbLogger forbids log_model=True with offline=True.
    This callback manually calls experiment.save() on the checkpoint directory.
    """
    def __init__(self, save_dir: str = None):
        self.save_dir = save_dir
        self.best_model_path = None

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        # Check if we have a wandb logger
        if trainer.logger and isinstance(trainer.logger, WandbLogger):
            # If checkpoint callback exists
            if trainer.checkpoint_callback:
                # We can save all files in dirpath
                self._save_checkpoints(trainer.logger, trainer.checkpoint_callback.dirpath)

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
         if trainer.logger and isinstance(trainer.logger, WandbLogger):
            if trainer.checkpoint_callback:
                self._save_checkpoints(trainer.logger, trainer.checkpoint_callback.dirpath)

    def _save_checkpoints(self, logger, dirpath):
        if not dirpath or not os.path.exists(dirpath):
            return
            
        # WandB 'save' with base_path argument preserves relative structure
        # We want to save everything in checkpoints dir
        # glob *.ckpt
        ckpt_files = glob.glob(os.path.join(dirpath, "*.ckpt"))
        for ckpt in ckpt_files:
            # Policy="now" ensures it's copied to wandb directory immediately (if offline)
            # or uploaded (if online)
            logger.experiment.save(ckpt, base_path=os.path.dirname(dirpath), policy="now")

        # Cleanup broken symlinks in the wandb directory
        # This is necessary because if a checkpoint is deleted by ModelCheckpoint (e.g. save_top_k),
        # the symlink in the wandb directory remains but points to a non-existent file.
        # This causes 'wandb sync' to fail.
        wandb_dir = logger.experiment.dir
        # Assuming base_path was os.path.dirname(dirpath) -> 'checkpoints' is the subdir
        wandb_ckpt_dir = os.path.join(wandb_dir, "checkpoints")
        
        if os.path.exists(wandb_ckpt_dir):
            for filename in os.listdir(wandb_ckpt_dir):
                filepath = os.path.join(wandb_ckpt_dir, filename)
                # Check if it is a broken link
                if os.path.islink(filepath) and not os.path.exists(filepath):
                    os.remove(filepath)
