import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Any, Dict, Tuple, Optional, Union

from src.models.components.spectrogram import Spectrogram
from src.models.components.masking import MaskingGenerator
from src.models.components.patch_embed import PatchEmbed
from src.models.components.vit import ViT
from src.utils.lr_schedulers import LinearWarmupCosineDecay

class AudioJEPAModule(L.LightningModule):
    """
    Audio-JEPA Lightning Module.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer configuration (partial).
        net (Dict[str, Any]): Configuration for sub-modules (spectrogram, patch_embed, masking, encoder, predictor).
        warmup_pct (float): Percentage of total steps for warmup.
        final_lr_ratio (float): Ratio of final learning rate to initial learning rate.
        ema_decay (float): Initial EMA decay rate.
        ema_end_decay (float): Final EMA decay rate.
        ema_anneal_end_step (int): Step at which EMA decay reaches ema_end_decay.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        net: Dict[str, Any],
        warmup_pct: float = 0.1,
        final_lr_ratio: float = 0.001,
        ema_decay: float = 0.996,
        ema_end_decay: float = 1.0,
        ema_anneal_end_step: Optional[int] = None,
        spectrogram_adjustment_mode: str = "pad",
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.warmup_pct = warmup_pct
        self.final_lr_ratio = final_lr_ratio
        self.spectrogram_adjustment_mode = spectrogram_adjustment_mode
        
        # Components
        self.spectrogram = Spectrogram(**net.get("spectrogram", {}))
        self.patch_embed = PatchEmbed(**net.get("patch_embed", {}))
        self.mask_generator = MaskingGenerator(**net.get("masking", {}))
        
        # Student (Encoder)
        self.student = ViT(**net.get("encoder", {}))
        
        # Teacher (Encoder) - same arch as student
        self.teacher = ViT(**net.get("encoder", {}))
        # Initialize teacher with student weights
        self.teacher.load_state_dict(self.student.state_dict())
        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        # Predictor
        predictor_config = net.get("predictor", {})
        self.predictor = ViT(**predictor_config)
        
        # Projections for Predictor
        encoder_dim = net.get("encoder", {}).get("embed_dim", 768)
        predictor_embed_dim = predictor_config.get("embed_dim", 768)
        
        self.predictor_input_proj = nn.Linear(encoder_dim, predictor_embed_dim)
        self.predictor_output_proj = nn.Linear(predictor_embed_dim, encoder_dim)
        
        # Mask Token
        # Mask token should match predictor dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # EMA parameters
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_end_step = ema_anneal_end_step
        self.current_ema_decay = ema_decay

    def setup(self, stage: Optional[str] = None) -> None:
        # Calculate ema_anneal_end_step if not provided
        if self.ema_anneal_end_step is None:
            if self.trainer.max_steps and self.trainer.max_steps > 0:
                self.ema_anneal_end_step = self.trainer.max_steps
            else:
                self.ema_anneal_end_step = self.trainer.estimated_stepping_batches
            
            # If still None or 0 (unlikely if trainer is set up correctly), default to something safe or warn
            if self.ema_anneal_end_step is None or self.ema_anneal_end_step <= 0:
                # Fallback to a large number to effectively disable annealing or just keep constant
                # But better to warn.
                print("Warning: Could not determine total steps for EMA annealing. Using 100000 as default.")
                self.ema_anneal_end_step = 100000
        
    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        # Update EMA decay: Linear schedule from start_decay to end_decay
        step = self.global_step
        decay = self.ema_end_decay - (self.ema_end_decay - self.ema_decay) * (
            (self.ema_anneal_end_step - step) / self.ema_anneal_end_step
        )
        decay = min(self.ema_end_decay, max(self.ema_decay, decay))
        self.current_ema_decay = decay
        
    def _update_teacher(self) -> None:
        with torch.no_grad():
            m = self.current_ema_decay
            for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.data)

    def _adjust_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Adjusts the spectrogram time dimension to be divisible by the patch size.
        
        Args:
            spec (torch.Tensor): Spectrogram [B, C, F, T].
            
        Returns:
            torch.Tensor: Adjusted spectrogram.
        """
        # Get patch size from PatchEmbed component
        # PatchEmbed stores patch_size as (H, W) corresponding to (F, T)
        patch_size = self.patch_embed.patch_embed.patch_size
        patch_time_dim = patch_size[1]
        
        T = spec.shape[-1]
        remainder = T % patch_time_dim
        
        if remainder != 0:
            if self.spectrogram_adjustment_mode == "pad":
                pad_amount = patch_time_dim - remainder
                spec = F.pad(spec, (0, pad_amount))
            elif self.spectrogram_adjustment_mode == "truncate":
                # Truncate the extra frames
                spec = spec[..., :T - remainder]
            else:
                raise ValueError(f"Unknown spectrogram_adjustment_mode: {self.spectrogram_adjustment_mode}")
            
        return spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference/eval. Returns student representation.
        """
        x = self.spectrogram(x)
        x = self._adjust_spectrogram(x)
        x = self.patch_embed(x)
        x = self.student(x)
        return x

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        waveform = batch["waveform"] # [B, 1, T]
        
        # 1. Spectrogram
        spec = self.spectrogram(waveform) # [B, 1, F, T]
        spec = self._adjust_spectrogram(spec)
        
        # 2. Patchify
        patches = self.patch_embed(spec) # [B, N, D]
        B, N, D = patches.shape
        
        # Calculate grid size for this batch
        patch_size = self.patch_embed.patch_embed.patch_size
        F_pix = spec.shape[2]
        T_pix = spec.shape[3]
        H_grid = F_pix // patch_size[0]
        W_grid = T_pix // patch_size[1]
        current_grid_size = (H_grid, W_grid)
        
        # 3. Masking
        # Generate ONE mask for the batch: [1, N]
        # We share the mask across the batch to enable vectorization
        mask = self.mask_generator(1, device=self.device, grid_size=current_grid_size) # [1, N]
        mask = mask.expand(B, -1) # [B, N]
        
        # Update teacher EMA
        self._update_teacher()
        
        # Teacher forward (on full image)
        with torch.no_grad():
            teacher_full = self.teacher(patches, grid_size=current_grid_size) # [B, N, D]
        
        # Vectorized processing
        # Since mask is same for all B, we can just take indices from the first one
        m = mask[0] # [N]
        keep_indices = torch.nonzero(~m).flatten() # [N_keep]
        mask_indices = torch.nonzero(m).flatten() # [N_mask]
        
        num_keep = len(keep_indices)
        num_mask = len(mask_indices)
        
        # Student input (Context)
        # patches: [B, N, D] -> select keep_indices -> [B, N_keep, D]
        context_patches = patches[:, keep_indices, :] 
        
        # Context Pos Ids
        # [N_keep] -> expand to [B, N_keep]
        context_pos_ids = keep_indices.unsqueeze(0).expand(B, -1)
        
        # Student forward
        student_out = self.student(
            context_patches, 
            pos_ids=context_pos_ids,
            grid_size=current_grid_size
        ) # [B, N_keep, D]
        
        # Predictor Input Construction
        # Project student output to predictor dimension
        # student_out: [B, N_keep, encoder_dim] -> [B, N_keep, predictor_dim]
        student_out_proj = self.predictor_input_proj(student_out)

        # Mask tokens: [1, 1, D] -> [B, N_mask, D]
        mask_tokens = self.mask_token.expand(B, num_mask, -1)
        
        if self.predictor.pos_embed_type != "rope":
            # Absolute pos embed
            # Add pos embed to mask tokens
            # pos_embed: [1, N, D] -> select mask_indices -> [1, N_mask, D] -> expand to [B, N_mask, D]
            mask_pos_embed = self.predictor.pos_embed[:, mask_indices, :].expand(B, -1, -1)
            mask_tokens = mask_tokens + mask_pos_embed
            
        pred_input = torch.cat([student_out_proj, mask_tokens], dim=1) # [B, N, D]
        
        # Reorder to original sequence order
        all_indices = torch.cat([keep_indices, mask_indices]) # [N]
        sort_indices = torch.argsort(all_indices) # [N]
        pred_input = pred_input[:, sort_indices, :] # [B, N, D]

        if self.predictor.pos_embed_type == "rope":
            # Since input is now ordered 0..N, we can rely on default pos_ids (arange)
            pred_out = self.predictor(
                pred_input, 
                pos_ids=None,
                grid_size=current_grid_size
            ) # [B, N, D]
        else:
            pred_out = self.predictor(pred_input, add_pos_embed=False) # [B, N, D]
            
        # Predictions for targets are at the mask indices
        # pred_out is now [B, N, D] in spatial order.
        # We gather the predictions at the mask locations.
        # mask_indices: [N_mask]
        # We need to gather from dimension 1.
        # expand mask_indices to [B, N_mask, D] for gather? 
        # Or just index since mask_indices shared?
        # pred_out[:, mask_indices, :] works efficiently
        predictions = pred_out[:, mask_indices, :] # [B, N_mask, predictor_dim]
        
        # Project back to encoder dimension to match targets
        predictions = self.predictor_output_proj(predictions) # [B, N_mask, encoder_dim]
        
        # Targets from Teacher
        # teacher_full: [B, N, D] -> select mask_indices -> [B, N_mask, D]
        teacher_targets = teacher_full[:, mask_indices, :]
        
        # Loss
        loss = F.mse_loss(predictions, teacher_targets)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        waveform = batch["waveform"]
        
        spec = self.spectrogram(waveform)
        spec = self._adjust_spectrogram(spec)
        patches = self.patch_embed(spec)
        B, N, D = patches.shape
        
        patch_size = self.patch_embed.patch_embed.patch_size
        F_pix = spec.shape[2]
        T_pix = spec.shape[3]
        H_grid = F_pix // patch_size[0]
        W_grid = T_pix // patch_size[1]
        current_grid_size = (H_grid, W_grid)
        
        mask = self.mask_generator(B, device=self.device, grid_size=current_grid_size)
        
        total_loss = 0.0
        
        # Teacher forward (full)
        with torch.no_grad():
            teacher_full = self.teacher(patches, grid_size=current_grid_size)
            
        for i in range(B):
            m = mask[i]
            p = patches[i]
            
            keep_indices = torch.nonzero(~m).flatten()
            mask_indices = torch.nonzero(m).flatten()
            
            context_patches = p[keep_indices]
            context_pos_ids = keep_indices
            
            student_out = self.student(
                context_patches.unsqueeze(0), 
                pos_ids=context_pos_ids.unsqueeze(0),
                grid_size=current_grid_size
            ).squeeze(0)
            
            num_targets = len(mask_indices)
            mask_tokens = self.mask_token.expand(1, num_targets, -1).squeeze(0)
            
            # Project student output
            student_out_proj = self.predictor_input_proj(student_out)
            
            if self.predictor.pos_embed_type != "rope":
                mask_pos_embed = self.predictor.pos_embed[:, mask_indices, :].squeeze(0)
                mask_tokens = mask_tokens + mask_pos_embed
            
            pred_input = torch.cat([student_out_proj, mask_tokens], dim=0)
            
            # Reorder
            all_indices = torch.cat([context_pos_ids, mask_indices], dim=0)
            sort_indices = torch.argsort(all_indices)
            pred_input = pred_input[sort_indices]
            
            if self.predictor.pos_embed_type == "rope":
                pred_out = self.predictor(
                    pred_input.unsqueeze(0), 
                    pos_ids=None,
                    grid_size=current_grid_size
                ).squeeze(0)
            else:
                pred_out = self.predictor(pred_input.unsqueeze(0), add_pos_embed=False).squeeze(0)
            
            # Extract predictions at mask indices
            predictions = pred_out[mask_indices]
            predictions = self.predictor_output_proj(predictions)
            teacher_targets = teacher_full[i, mask_indices, :]
            
            loss = F.mse_loss(predictions, teacher_targets)
            total_loss += loss
            
        avg_loss = total_loss / B
        self.log("val/loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        return avg_loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        
        # Determine total steps
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            total_steps = self.trainer.max_steps
        else:
            total_steps = self.trainer.estimated_stepping_batches
            
        warmup_steps = int(total_steps * self.warmup_pct)
        
        lr_lambda = LinearWarmupCosineDecay(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            final_lr_ratio=self.final_lr_ratio
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
