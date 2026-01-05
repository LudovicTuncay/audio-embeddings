import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import vector_norm
from typing import Any, Dict, Optional, Tuple

from src.models.audio_jepa_module import AudioJEPAModule
from src.models.components.random_projection_quantizer import RandomProjectionQuantizer

class RQJEPAModule(AudioJEPAModule):
    """
    RQ-JEPA Lightning Module.
    Extends AudioJEPAModule with Random Projection Quantization loss.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer configuration.
        net (Dict[str, Any]): Configuration for sub-modules.
        warmup_pct (float): Percentage of total steps for warmup.
        final_lr_ratio (float): Ratio of final learning rate to initial learning rate.
        ema_decay (float): Initial EMA decay rate.
        ema_end_decay (float): Final EMA decay rate.
        ema_anneal_end_step (int): Step at which EMA decay reaches ema_end_decay.
        spectrogram_adjustment_mode (str): 'pad' or 'truncate' for spectrogram time dimension.
        jepa_criterion (torch.nn.Module): Loss function for JEPA (defaults to MSELoss).
        rq_criterion (torch.nn.Module): Loss function for RQ (defaults to CrossEntropyLoss).
        rq_lambda (float): Weight for JEPA loss (1 - rq_lambda is used for RQ loss).
        codebook_dim (int): Codebook dimension for RandomProjectionQuantizer.
        vocab_size (int): Vocabulary size for RandomProjectionQuantizer.
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
        jepa_criterion: Optional[torch.nn.Module] = None,
        rq_criterion: Optional[torch.nn.Module] = None,
        rq_lambda: float = 0.5,
        codebook_dim: int = 16,
        vocab_size: int = 8192,
    ):
        super().__init__(
            optimizer=optimizer,
            net=net,
            warmup_pct=warmup_pct,
            final_lr_ratio=final_lr_ratio,
            ema_decay=ema_decay,
            ema_end_decay=ema_end_decay,
            ema_anneal_end_step=ema_anneal_end_step,
            spectrogram_adjustment_mode=spectrogram_adjustment_mode,
            criterion=jepa_criterion, # Pass jepa_criterion as criterion to base class
        )
        self.save_hyperparameters(logger=False, ignore=["jepa_criterion", "rq_criterion"])
        
        self.rq_lambda = rq_lambda
        # Store rq_criterion separately
        self.rq_criterion = rq_criterion if rq_criterion is not None else nn.CrossEntropyLoss()
        
        # Random Projection Quantizer
        # Input to quantizer is teacher output which has encoder_dim
        encoder_dim = net.get("encoder", {}).get("embed_dim", 768)
        self.quantizer = RandomProjectionQuantizer(
            input_dim=encoder_dim,
            cb_dim=codebook_dim,
            cb_vocab=vocab_size
        )
        # Freeze quantizer (it is random and fixed)
        for p in self.quantizer.parameters():
            p.requires_grad = False
            
        # Projection head for RQ prediction
        # Takes predictor output (pred_dim) and predicts code indices (vocab_size)
        predictor_config = net.get("predictor", {})
        predictor_embed_dim = predictor_config.get("embed_dim", 768)
        self.rq_proj = nn.Linear(predictor_embed_dim, vocab_size)

    def _calculate_combined_loss(
        self, 
        predictions_raw: torch.Tensor, 
        teacher_targets: torch.Tensor, 
        rq_logits: torch.Tensor,
        rq_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates both JEPA and RQ losses and combines them.
        """
        # --- JEPA Loss ---
        # Project back to encoder dimension for JEPA loss
        predictions_jepa = self.predictor_output_proj(predictions_raw) # [B, N_mask, encoder_dim]
        
        jepa_loss = self.criterion(predictions_jepa, teacher_targets) # Uses self.criterion (mapped from jepa_criterion)
        
        # --- RQ Loss ---
        # Calculate Scale RQ Loss
        # Flatten for loss calculation
        rq_loss = self.rq_criterion(
            rq_logits.reshape(-1, self.hparams.vocab_size), 
            rq_targets.reshape(-1)
        )
        
        # --- Combine ---
        total_loss = self.rq_lambda * jepa_loss + (1 - self.rq_lambda) * rq_loss
        
        return total_loss, jepa_loss, rq_loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        waveform = batch["waveform"]
        
        patches, current_grid_size = self._process_audio(waveform)
        B, N, D = patches.shape
        
        mask = self.mask_generator(1, device=self.device, grid_size=current_grid_size)
        mask = mask.expand(B, -1)
        
        student_out = self.compute_student(patches, mask, current_grid_size)
        predictions_raw = self.compute_predictor(student_out, mask, current_grid_size)
        
        self._update_teacher()
        
        with torch.no_grad():
            teacher_full = self.teacher(patches, grid_size=current_grid_size)
        
        # Prepare targets and logits for RQ-JEPA
        m = mask[0]
        mask_indices = torch.nonzero(m).flatten()
        
        # Teacher targets at masked locations
        teacher_targets = teacher_full[:, mask_indices, :] # [B, N_mask, encoder_dim]

        # RQ Targets (Quantized)
        with torch.no_grad():
            rq_targets = self.quantizer(teacher_targets) # [B, N_mask]

        # RQ Logits
        rq_logits = self.rq_proj(predictions_raw) # [B, N_mask, vocab_size]

        loss, jepa_loss, rq_loss = self._calculate_combined_loss(predictions_raw, teacher_targets, rq_logits, rq_targets)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("train/jepa_loss", jepa_loss, on_step=True, on_epoch=True, batch_size=B)
        self.log("train/rq_loss", rq_loss, on_step=True, on_epoch=True, batch_size=B)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        waveform = batch["waveform"]
        
        patches, current_grid_size = self._process_audio(waveform)
        B, N, D = patches.shape
        
        mask = self.mask_generator(1, device=self.device, grid_size=current_grid_size)
        mask = mask.expand(B, -1)
        
        student_out = self.compute_student(patches, mask, current_grid_size)
        predictions_raw = self.compute_predictor(student_out, mask, current_grid_size)
        
        with torch.no_grad():
            teacher_full = self.teacher(patches, grid_size=current_grid_size)
            
            # Prepare targets and logits for RQ-JEPA
            m = mask[0]
            mask_indices = torch.nonzero(m).flatten()
            
            # Teacher targets at masked locations
            teacher_targets = teacher_full[:, mask_indices, :] # [B, N_mask, encoder_dim]

            # RQ Targets (Quantized)
            rq_targets = self.quantizer(teacher_targets) # [B, N_mask]
            
            # RQ Logits
            rq_logits = self.rq_proj(predictions_raw) # [B, N_mask, vocab_size]
            
            loss, jepa_loss, rq_loss = self._calculate_combined_loss(predictions_raw, teacher_targets, rq_logits, rq_targets)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("val/jepa_loss", jepa_loss, on_step=False, on_epoch=True, batch_size=B)
        self.log("val/rq_loss", rq_loss, on_step=False, on_epoch=True, batch_size=B)
        return loss
