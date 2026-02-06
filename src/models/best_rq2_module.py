import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import lightning as L
from typing import Any, Dict, Tuple, Optional

from src.models.components.spectrogram import Spectrogram
from src.models.components.masking import MaskingGenerator
from src.models.components.patch_embed import PatchEmbed
from src.models.components.vit import ViT
from src.models.components.random_projection_quantizer import RandomProjectionQuantizer
from src.utils.lr_schedulers import LinearWarmupCosineDecay


class BestRQ2Module(L.LightningModule):
    """
    Best-RQ 2 Lightning Module.

    Implements a 2-step (Encoder-Predictor) Masked Audio Modeling approach using
    Random Projection Quantization of spectrogram patches as targets.
    Equivalent to RQA-JEPA with lambda=0 and rq_input_type='spectrogram',
    but optimized to remove the Teacher model entirely.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer configuration.
        net (Dict[str, Any]): Configuration for sub-modules.
        warmup_pct (float): Percentage of total steps for warmup.
        final_lr_ratio (float): Ratio of final learning rate to initial learning rate.
        spectrogram_adjustment_mode (str): 'pad' or 'truncate' for spectrogram time dimension.
        codebook_dim (int): Codebook dimension for RandomProjectionQuantizer.
        vocab_size (int): Vocabulary size for RandomProjectionQuantizer.
        criterion (torch.nn.Module): Loss function (defaults to CrossEntropyLoss).
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
        criterion: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            logger=False, ignore=["criterion", "net", "optimizer"]
        )

        self.warmup_pct = warmup_pct
        self.final_lr_ratio = final_lr_ratio
        self.spectrogram_adjustment_mode = spectrogram_adjustment_mode
        self.vocab_size = vocab_size

        # Optimizer partial
        self.optimizer_config = optimizer

        # Loss
        if criterion is not None:
            self.criterion = (
                criterion()
                if isinstance(criterion, (type, functools.partial))
                or (callable(criterion) and not isinstance(criterion, nn.Module))
                else criterion
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Components
        self.spectrogram = Spectrogram(**net.get("spectrogram", {}))
        self.patch_embed = PatchEmbed(**net.get("patch_embed", {}))
        self.mask_generator = MaskingGenerator(**net.get("masking", {}))

        # Encoder
        self.encoder = ViT(**net.get("encoder", {}))

        # Predictor
        predictor_config = net.get("predictor", {})
        self.predictor = ViT(**predictor_config)

        # Dimensions
        encoder_dim = net.get("encoder", {}).get("embed_dim", 768)
        predictor_embed_dim = predictor_config.get("embed_dim", 768)

        # Adapter: Encoder -> Predictor
        self.predictor_input_proj = nn.Linear(encoder_dim, predictor_embed_dim)

        # Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
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

        # Output Projection: Predictor -> Vocab
        self.rq_proj = nn.Linear(predictor_embed_dim, vocab_size)

    def _adjust_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
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
        spec = self.spectrogram(waveform)  # [B, 1, F, T]
        spec = self._adjust_spectrogram(spec)
        patches = self.patch_embed(spec)  # [B, N, D]

        patch_size = self.patch_embed.patch_embed.patch_size
        F_pix = spec.shape[2]
        T_pix = spec.shape[3]
        H_grid = F_pix // patch_size[0]
        W_grid = T_pix // patch_size[1]
        grid_size = (H_grid, W_grid)

        return patches, grid_size

    def _get_raw_patches(self, spec: torch.Tensor) -> torch.Tensor:
        """Extract raw key-value patches from spectrogram."""
        patch_size = self.patch_embed.patch_size  # (H, W)
        # Using kernel_size=patch_size, stride=patch_size ensures non-overlapping patches
        patches = F.unfold(spec, kernel_size=patch_size, stride=patch_size)  # [B, D, N]
        patches = patches.transpose(1, 2)  # [B, N, D]
        return patches

    def compute_encoder(
        self, patches: torch.Tensor, mask: torch.Tensor, grid_size: Tuple[int, int]
    ) -> torch.Tensor:
        B, N, _ = patches.shape
        m = mask[0]  # [N]
        keep_indices = torch.nonzero(~m).flatten()  # [N_keep]

        context_patches = patches[:, keep_indices, :]  # [B, N_keep, D]
        context_pos_ids = keep_indices.unsqueeze(0).expand(B, -1)  # [B, N_keep]

        encoder_out = self.encoder(
            context_patches, pos_ids=context_pos_ids, grid_size=grid_size
        )
        return encoder_out

    def compute_predictor(
        self, encoder_out: torch.Tensor, mask: torch.Tensor, grid_size: Tuple[int, int]
    ) -> torch.Tensor:
        B, N_keep, _ = encoder_out.shape
        m = mask[0]
        keep_indices = torch.nonzero(~m).flatten()
        mask_indices = torch.nonzero(m).flatten()
        num_mask = len(mask_indices)

        encoder_out_proj = self.predictor_input_proj(
            encoder_out
        )  # [B, N_keep, pred_dim]
        mask_tokens = self.mask_token.expand(B, num_mask, -1)

        if self.predictor.pos_embed_type != "rope":
            mask_pos_embed = self.predictor.pos_embed[:, mask_indices, :].expand(
                B, -1, -1
            )
            mask_tokens = mask_tokens + mask_pos_embed

        pred_input = torch.cat([encoder_out_proj, mask_tokens], dim=1)

        all_indices = torch.cat([keep_indices, mask_indices])
        sort_indices = torch.argsort(all_indices)
        pred_input = pred_input[:, sort_indices, :]

        if self.predictor.pos_embed_type == "rope":
            pred_out = self.predictor(pred_input, pos_ids=None, grid_size=grid_size)
        else:
            pred_out = self.predictor(pred_input, add_pos_embed=False)

        predictions_raw = pred_out[:, mask_indices, :]  # [B, N_mask, pred_dim]
        return predictions_raw

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        waveform = batch["waveform"]

        # 1. Process Audio
        patches, current_grid_size = self._process_audio(waveform)
        B, N, D = patches.shape

        # 2. Masking
        mask = self.mask_generator(1, device=self.device, grid_size=current_grid_size)
        mask = mask.expand(B, -1)

        # 3. Targets (Best-RQ: Quantized Raw Patches)
        with torch.no_grad():
            spec = self.spectrogram(waveform)
            spec = self._adjust_spectrogram(spec)
            raw_patches = self._get_raw_patches(spec)

            m = mask[0]
            mask_indices = torch.nonzero(m).flatten()

            target_input = raw_patches[:, mask_indices, :]
            targets = self.quantizer(target_input)  # [B, N_mask]

        # 4. Predictions (Encoder -> Predictor -> Proj)
        encoder_out = self.compute_encoder(patches, mask, current_grid_size)
        predictions_raw = self.compute_predictor(encoder_out, mask, current_grid_size)
        logits = self.rq_proj(predictions_raw)  # [B, N_mask, vocab_size]

        # 5. Loss
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

        with torch.no_grad():
            spec = self.spectrogram(waveform)
            spec = self._adjust_spectrogram(spec)
            raw_patches = self._get_raw_patches(spec)

            m = mask[0]
            mask_indices = torch.nonzero(m).flatten()

            target_input = raw_patches[:, mask_indices, :]
            targets = self.quantizer(target_input)

        encoder_out = self.compute_encoder(patches, mask, current_grid_size)
        predictions_raw = self.compute_predictor(encoder_out, mask, current_grid_size)
        logits = self.rq_proj(predictions_raw)

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
