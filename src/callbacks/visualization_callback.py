import torch
import matplotlib.pyplot as plt
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from typing import Any, Dict, Optional

class VisualizationCallback(Callback):
    """
    Callback to visualize spectrograms, patches, and masks.
    Logs the first 4 samples of the first 2 batches.
    """
    def __init__(self, num_samples: int = 4):
        super().__init__()
        self.num_samples = num_samples
        self.batches_logged = 0

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.batches_logged >= 2:
            return
            
        # Log for the first 2 batches
        if batch_idx < 2:
            self._log_visualizations(trainer, pl_module, batch, batch_idx)
            self.batches_logged += 1
            
    def _log_visualizations(self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Dict[str, Any], batch_idx: int) -> None:
        logger = trainer.logger
        if not isinstance(logger, WandbLogger):
            return
            
        waveform = batch["waveform"][:self.num_samples] # [B, 1, T]
        
        # Get sample rate dynamically
        sample_rate = 32000 # Default
        if hasattr(pl_module, "spectrogram") and hasattr(pl_module.spectrogram, "sample_rate"):
            sample_rate = pl_module.spectrogram.sample_rate
        elif hasattr(pl_module, "hparams") and "net" in pl_module.hparams and "spectrogram" in pl_module.hparams.net:
             sample_rate = pl_module.hparams.net["spectrogram"].get("sample_rate", 32000)
        
        # Get spectrograms
        with torch.no_grad():
            spec = pl_module.spectrogram(waveform.to(pl_module.device)) # [B, 1, F, T]
            
            # Get grid size and patch info
            patch_size = pl_module.patch_embed.patch_embed.patch_size
            F_pix = spec.shape[2]
            T_pix = spec.shape[3]
            H_grid = F_pix // patch_size[0]
            W_grid = T_pix // patch_size[1]
            current_grid_size = (H_grid, W_grid)
            
            # Generate mask
            # Using the same logic as training step (shared mask across batch)
            # But we want to see if it's the same across batches (it should be random each step)
            mask = pl_module.mask_generator(1, device=pl_module.device, grid_size=current_grid_size) # [1, N]
            mask = mask.expand(self.num_samples, -1) # [B, N]
            
        # Log to WandB
        import wandb
        
        columns = ["Batch Idx", "Sample Idx", "Audio", "Spectrogram", "Masked Spectrogram (Context)", "Inverse Masked Spectrogram (Targets)"]
        data = []
        
        for i in range(self.num_samples):
            # Audio
            audio_data = waveform[i].squeeze().cpu().numpy()
            audio = wandb.Audio(audio_data, sample_rate=sample_rate, caption=f"B{batch_idx}_S{i}")
            
            # Spectrograms
            spec_data = spec[i].squeeze().cpu().numpy()
            mask_data = mask[i].cpu().numpy()
            
            # 1. Original
            fig_orig = self._plot_spectrogram(spec_data, patch_size, current_grid_size)
            img_orig = wandb.Image(fig_orig, caption=f"Spec B{batch_idx}_S{i}")
            plt.close(fig_orig)
            
            # 2. Masked (Context) - Masked parts are dark
            fig_masked = self._plot_spectrogram_with_mask(spec_data, mask_data, patch_size, current_grid_size, invert_mask=False)
            img_masked = wandb.Image(fig_masked, caption=f"Masked B{batch_idx}_S{i}")
            plt.close(fig_masked)
            
            # 3. Inverse Masked (Targets) - Context parts are dark
            fig_inv_masked = self._plot_spectrogram_with_mask(spec_data, mask_data, patch_size, current_grid_size, invert_mask=True)
            img_inv_masked = wandb.Image(fig_inv_masked, caption=f"InvMasked B{batch_idx}_S{i}")
            plt.close(fig_inv_masked)
            
            data.append([batch_idx, i, audio, img_orig, img_masked, img_inv_masked])
            
        # Log Table
        table = wandb.Table(columns=columns, data=data)
        logger.experiment.log({f"train/visualizations_batch_{batch_idx}": table})
            
    def _plot_spectrogram(
        self, 
        spec: np.ndarray, 
        patch_size: tuple[int, int], 
        grid_size: tuple[int, int]
    ) -> plt.Figure:
        """Plots spectrogram with grid lines."""
        return self._plot_spectrogram_with_mask(spec, None, patch_size, grid_size)

    def _plot_spectrogram_with_mask(
        self, 
        spec: np.ndarray, 
        mask: Optional[np.ndarray], 
        patch_size: tuple[int, int], 
        grid_size: tuple[int, int],
        invert_mask: bool = False
    ) -> plt.Figure:
        """
        Plots spectrogram with dashed grid lines and darker masked patches.
        If mask is None, just plots spectrogram and grid.
        If invert_mask is True, darkens the unmasked parts instead.
        """
        H_grid, W_grid = grid_size
        Ph, Pw = patch_size
        H, W = spec.shape
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(spec, origin="lower", aspect="auto", cmap="viridis")
        
        # Overlay Grid
        for h in range(0, H + 1, Ph):
            ax.axhline(h - 0.5, color="white", linestyle="--", linewidth=0.5, alpha=0.5)
        for w in range(0, W + 1, Pw):
            ax.axvline(w - 0.5, color="white", linestyle="--", linewidth=0.5, alpha=0.5)
            
        # Overlay Mask
        if mask is not None:
            mask_grid = mask.reshape(H_grid, W_grid)
            if invert_mask:
                mask_grid = ~mask_grid
            
            overlay = np.zeros((H, W, 4)) # RGBA
            for r in range(H_grid):
                for c in range(W_grid):
                    if mask_grid[r, c]:
                        y_start = r * Ph
                        y_end = (r + 1) * Ph
                        x_start = c * Pw
                        x_end = (c + 1) * Pw
                        overlay[y_start:y_end, x_start:x_end, 3] = 0.7 
                        
            ax.imshow(overlay, origin="lower", aspect="auto")
            
        ax.set_title("Spectrogram")
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Frequency Bins")
        plt.tight_layout()
        return fig
