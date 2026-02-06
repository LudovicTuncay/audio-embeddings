import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Any, Dict, Tuple

from src.models.components.spectrogram import Spectrogram
from src.models.components.masking import MaskingGenerator
from src.models.components.patch_embed import PatchEmbed
from src.models.components.vit import ViT
from src.models.components.random_projection_quantizer import RandomProjectionQuantizer
from src.utils.lr_schedulers import LinearWarmupCosineDecay


class BestRQModule(L.LightningModule):
    """
    Best-RQ Lightning Module.

    Implements a single-stage Masked Audio Modeling approach using Random Projection Quantization targets.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer configuration.
        net (Dict[str, Any]): Configuration for sub-modules.
        warmup_pct (float): Percentage of total steps for warmup.
        final_lr_ratio (float): Ratio of final learning rate to initial learning rate.
        spectrogram_adjustment_mode (str): 'pad' or 'truncate' for spectrogram time dimension.
        codebook_dim (int): Codebook dimension for RandomProjectionQuantizer.
        vocab_size (int): Vocabulary size for RandomProjectionQuantizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        net: Dict[str, Any],
        warmup_pct: float = 0.1,
        final_lr_ratio: float = 0.001,
        spectrogram_adjustment_mode: str = "pad",
        codebook_dim: int = 16,
        vocab_size: int = 8192,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net", "optimizer"])

        self.warmup_pct = warmup_pct
        self.final_lr_ratio = final_lr_ratio
        self.spectrogram_adjustment_mode = spectrogram_adjustment_mode
        self.vocab_size = vocab_size

        # Store optimizer partial
        self.optimizer_config = optimizer

        # Components
        self.spectrogram = Spectrogram(**net.get("spectrogram", {}))
        self.patch_embed = PatchEmbed(**net.get("patch_embed", {}))
        self.mask_generator = MaskingGenerator(**net.get("masking", {}))

        # Encoder (ViT)
        self.encoder = ViT(**net.get("encoder", {}))

        # Mask Token
        encoder_dim = net.get("encoder", {}).get("embed_dim", 768)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Random Projection Quantizer
        # Input to quantizer is raw patches
        patch_size = self.patch_embed.patch_size
        in_chans = self.patch_embed.in_chans
        quantizer_input_dim = patch_size[0] * patch_size[1] * in_chans

        self.quantizer = RandomProjectionQuantizer(
            input_dim=quantizer_input_dim, cb_dim=codebook_dim, cb_vocab=vocab_size
        )
        # Freeze quantizer
        for p in self.quantizer.parameters():
            p.requires_grad = False

        # Projection head
        self.output_proj = nn.Linear(encoder_dim, vocab_size)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

    def _adjust_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Adjusts the spectrogram time dimension to be divisible by the patch size.
        """
        patch_size = self.patch_embed.patch_embed.patch_size
        patch_time_dim = patch_size[1]

        T = spec.shape[-1]
        remainder = T % patch_time_dim

        if remainder != 0:
            if self.spectrogram_adjustment_mode == "pad":
                pad_amount = patch_time_dim - remainder
                spec = F.pad(spec, (0, pad_amount))
            elif self.spectrogram_adjustment_mode == "truncate":
                spec = spec[..., : T - remainder]
            else:
                raise ValueError(
                    f"Unknown spectrogram_adjustment_mode: {self.spectrogram_adjustment_mode}"
                )

        return spec

    def _process_audio(
        self, waveform: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Processes raw waveform into patches and returns patches and grid size.
        """
        # 1. Spectrogram
        spec = self.spectrogram(waveform)  # [B, 1, F, T]
        spec = self._adjust_spectrogram(spec)

        # 2. Patchify
        patches = self.patch_embed(spec)  # [B, N, D]

        # Calculate grid size
        patch_size = self.patch_embed.patch_embed.patch_size
        F_pix = spec.shape[2]
        T_pix = spec.shape[3]
        H_grid = F_pix // patch_size[0]
        W_grid = T_pix // patch_size[1]
        grid_size = (H_grid, W_grid)

        return patches, grid_size

    def _get_raw_patches(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Extract raw key-value patches from spectrogram for quantization.
        """
        patch_size = self.patch_embed.patch_size  # (H, W)

        # F.unfold returns [B, C*pH*pW, L]
        # Spectrogram is [B, C, F, T]
        # patch_size is (H, W) -> (freq_patch, time_patch)

        patches = F.unfold(spec, kernel_size=patch_size, stride=patch_size)  # [B, D, N]
        patches = patches.transpose(1, 2)  # [B, N, D]

        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference/eval. Returns encoder representation.
        """
        patches, grid_size = self._process_audio(x)
        x = self.encoder(patches, grid_size=grid_size)
        return x

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        waveform = batch["waveform"]

        # 1. Process Audio
        patches, current_grid_size = self._process_audio(waveform)
        B, N, D = patches.shape

        # 2. Generate Mask
        mask = self.mask_generator(1, device=self.device, grid_size=current_grid_size)
        mask = mask.expand(B, -1)  # [B, N]

        # 3. Prepare Inputs (Encoder sees full sequence with mask tokens)
        encoder_input = patches.clone()
        mask_tokens_expanded = self.mask_token.expand(B, N, -1)

        # Replace masked patches with mask tokens
        mask_bool = mask.bool()  # [B, N]
        encoder_input[mask_bool] = mask_tokens_expanded[mask_bool]

        # 4. Encoder Forward
        # We pass the full sequence.
        # For RoPE, pos_ids are auto-generated as 0..N-1 if None, which matches the grid layout.
        encoder_out = self.encoder(
            encoder_input, grid_size=current_grid_size
        )  # [B, N, D]

        # 5. Get Targets (Quantized Raw Patches)
        with torch.no_grad():
            # Re-compute spec for raw patches
            spec = self.spectrogram(waveform)
            spec = self._adjust_spectrogram(spec)
            raw_patches = self._get_raw_patches(spec)  # [B, N, raw_dim]

            # Select masked patches for targets
            m = mask[0]
            mask_indices = torch.nonzero(m).flatten()

            target_input = raw_patches[:, mask_indices, :]  # [B, N_mask, raw_dim]
            targets = self.quantizer(target_input)  # [B, N_mask]

        # 6. Get Predictions
        # Select masked outputs
        predictions = encoder_out[:, mask_indices, :]  # [B, N_mask, D]
        logits = self.output_proj(predictions)  # [B, N_mask, vocab_size]

        # 7. Loss
        loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        waveform = batch["waveform"]

        patches, current_grid_size = self._process_audio(waveform)
        B, N, D = patches.shape

        mask = self.mask_generator(1, device=self.device, grid_size=current_grid_size)
        mask = mask.expand(B, -1)

        encoder_input = patches.clone()
        mask_tokens_expanded = self.mask_token.expand(B, N, -1)
        mask_bool = mask.bool()
        encoder_input[mask_bool] = mask_tokens_expanded[mask_bool]

        encoder_out = self.encoder(encoder_input, grid_size=current_grid_size)

        with torch.no_grad():
            spec = self.spectrogram(waveform)
            spec = self._adjust_spectrogram(spec)
            raw_patches = self._get_raw_patches(spec)

            m = mask[0]
            mask_indices = torch.nonzero(m).flatten()

            target_input = raw_patches[:, mask_indices, :]
            targets = self.quantizer(target_input)

        predictions = encoder_out[:, mask_indices, :]
        logits = self.output_proj(predictions)

        loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=B
        )
        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.optimizer_config(params=self.parameters())

        if self.trainer.max_steps and self.trainer.max_steps > 0:
            total_steps = self.trainer.max_steps
        else:
            total_steps = self.trainer.estimated_stepping_batches

        warmup_steps = int(total_steps * self.warmup_pct)

        lr_lambda = LinearWarmupCosineDecay(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            final_lr_ratio=self.final_lr_ratio,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "step",
                "frequency": 1,
            },
        }
