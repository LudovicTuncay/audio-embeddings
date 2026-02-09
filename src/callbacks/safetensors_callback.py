import os
import glob
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from lightning_utilities.core.rank_zero import rank_zero_warn
from safetensors.torch import save_file


class SafetensorsCallback(Callback):
    """
    Callback to save a corresponding .safetensors file whenever a .ckpt is saved.
    This allows for safe sharing of weights while keeping the full .ckpt (with optimizer state)
    local for resuming training.
    """

    def __init__(self, cleanup_orphan_safetensors: bool = False) -> None:
        self.cleanup_orphan_safetensors = cleanup_orphan_safetensors

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.checkpoint_callback:
            self._convert_checkpoints(trainer.checkpoint_callback.dirpath)

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        if trainer.checkpoint_callback:
            self._convert_checkpoints(trainer.checkpoint_callback.dirpath)

    def _convert_checkpoints(self, dirpath):
        if not dirpath or not os.path.exists(dirpath):
            return

        # Find all .ckpt files
        ckpt_files = glob.glob(os.path.join(dirpath, "*.ckpt"))
        ckpt_stems = {
            os.path.splitext(os.path.basename(ckpt_path))[0] for ckpt_path in ckpt_files
        }

        if self.cleanup_orphan_safetensors:
            safetensors_files = glob.glob(os.path.join(dirpath, "*.safetensors"))
            for safetensors_path in safetensors_files:
                base_name = os.path.splitext(os.path.basename(safetensors_path))[0]
                if base_name not in ckpt_stems:
                    try:
                        os.remove(safetensors_path)
                    except OSError as exc:
                        rank_zero_warn(
                            f"Failed to remove orphan safetensors file {safetensors_path}: {exc}"
                        )

        for ckpt_path in ckpt_files:
            # Construct safetensors path
            base_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            sf_path = os.path.join(dirpath, f"{base_name}.safetensors")

            # Check if we should convert:
            # 1. If safetensors doesn't exist
            # 2. Or if ckpt is newer than safetensors (e.g. last.ckpt was updated)
            should_convert = False
            if not os.path.exists(sf_path):
                should_convert = True
            else:
                if os.path.getmtime(ckpt_path) > os.path.getmtime(sf_path):
                    should_convert = True

            if should_convert:
                try:
                    # Load checkpoint (CPU)
                    # We accept "unsafe" load here because we created these files locally
                    checkpoint = torch.load(
                        ckpt_path, map_location="cpu", weights_only=False
                    )

                    # Extract state_dict
                    if "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint

                    # Filter out non-tensor values just in case
                    clean_state_dict = {
                        k: v
                        for k, v in state_dict.items()
                        if isinstance(v, torch.Tensor)
                    }

                    # Save as safetensors
                    save_file(clean_state_dict, sf_path)

                except Exception as e:
                    rank_zero_warn(f"Failed to convert {ckpt_path} to safetensors: {e}")
