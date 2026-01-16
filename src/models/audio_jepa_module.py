import torch
import math
import functools
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
        spectrogram_adjustment_mode (str): 'pad' or 'truncate' for spectrogram time dimension.
        criterion (torch.nn.Module): Loss function (defaults to MSELoss).
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
        criterion: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["criterion", "net", "optimizer"])
        
        self.warmup_pct = warmup_pct
        self.final_lr_ratio = final_lr_ratio
        self.spectrogram_adjustment_mode = spectrogram_adjustment_mode
        
        # Handle Criterion (support partials/factories to avoid checkpointing warnings)
        if criterion is not None:
            self.criterion = criterion() if isinstance(criterion, (type, functools.partial)) or callable(criterion) and not isinstance(criterion, nn.Module) else criterion
        else:
            self.criterion = nn.MSELoss()
            
        # Store optimizer partial to avoid saving it in hparams
        self.optimizer_config = optimizer
        
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
        # stop gradient (teacher will be updated by EMA)
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
            self.ema_anneal_end_step = getattr(self.trainer, "max_steps", 0)
            if self.ema_anneal_end_step <= 0:
                self.ema_anneal_end_step = getattr(self.trainer, "estimated_stepping_batches", 100000)
            
            if self.ema_anneal_end_step <= 0:
                print("Warning: Could not determine total steps for EMA annealing. Using 100000 as default.")
                self.ema_anneal_end_step = 100000
        
    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        # Update EMA decay
        step = self.global_step
        progress = (self.ema_anneal_end_step - step) / self.ema_anneal_end_step
        decay = self.ema_end_decay - (self.ema_end_decay - self.ema_decay) * progress
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
                spec = spec[..., :T - remainder]
            else:
                raise ValueError(f"Unknown spectrogram_adjustment_mode: {self.spectrogram_adjustment_mode}")
            
        return spec

    def _process_audio(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Processes raw waveform into patches and returns patches and grid size.
        
        Returns:
            patches: [B, N, D]
            grid_size: (H, W)
        """
        # 1. Spectrogram
        spec = self.spectrogram(waveform) # [B, 1, F, T]
        spec = self._adjust_spectrogram(spec)
        
        # 2. Patchify
        patches = self.patch_embed(spec) # [B, N, D]
        
        # Calculate grid size
        patch_size = self.patch_embed.patch_embed.patch_size
        F_pix = spec.shape[2]
        T_pix = spec.shape[3]
        H_grid = F_pix // patch_size[0]
        W_grid = T_pix // patch_size[1]
        grid_size = (H_grid, W_grid)
        
        return patches, grid_size

    def compute_student(self, patches: torch.Tensor, mask: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        """
        Computes the student output for unmasked patches.

        Args:
            patches: [B, N, D]
            mask: [B, N]
            grid_size: (H, W)

        Returns:
            student_out: [B, N_keep, D]
        """
        B, N, _ = patches.shape
        
        m = mask[0] # [N]
        keep_indices = torch.nonzero(~m).flatten() # [N_keep]
        
        # Student input (Context)
        context_patches = patches[:, keep_indices, :] # [B, N_keep, D]
        
        # Context Pos Ids
        context_pos_ids = keep_indices.unsqueeze(0).expand(B, -1) # [B, N_keep]
        
        # Student forward
        student_out = self.student(
            context_patches, 
            pos_ids=context_pos_ids,
            grid_size=grid_size
        ) # [B, N_keep, D]
        
        return student_out

    def compute_predictor(self, student_out: torch.Tensor, mask: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        """
        Computes the predictor output at masked locations.

        Args:
            student_out: [B, N_keep, D]
            mask: [B, N]
            grid_size: (H, W)

        Returns:
            predictions_raw: [B, N_mask, pred_dim]
        """
        B, N_keep, _ = student_out.shape
        # Note: B derived from student_out might be different if batch size changes, but it shouldn't here.
        # N is implicit in mask.
        
        m = mask[0] # [N]
        keep_indices = torch.nonzero(~m).flatten() # [N_keep]
        mask_indices = torch.nonzero(m).flatten() # [N_mask]
        num_mask = len(mask_indices)
        
        # Predictor Input Construction
        student_out_proj = self.predictor_input_proj(student_out) # [B, N_keep, pred_dim]

        # Mask tokens: [1, 1, pred_dim] -> [B, N_mask, pred_dim]
        mask_tokens = self.mask_token.expand(B, num_mask, -1)
        
        if self.predictor.pos_embed_type != "rope":
            # Absolute pos embed added to mask tokens
            mask_pos_embed = self.predictor.pos_embed[:, mask_indices, :].expand(B, -1, -1)
            mask_tokens = mask_tokens + mask_pos_embed
            
        pred_input = torch.cat([student_out_proj, mask_tokens], dim=1) # [B, N, pred_dim]
        
        # Reorder to original sequence order
        all_indices = torch.cat([keep_indices, mask_indices]) # [N]
        sort_indices = torch.argsort(all_indices) # [N]
        pred_input = pred_input[:, sort_indices, :] # [B, N, pred_dim]

        if self.predictor.pos_embed_type == "rope":
            # Rope handles positions internally if full sequence is provided
            pred_out = self.predictor(
                pred_input, 
                pos_ids=None,
                grid_size=grid_size
            )
        else:
            pred_out = self.predictor(pred_input, add_pos_embed=False)
            
        # Predictions at mask locations (returns raw embeddings in pred_dim)
        predictions_raw = pred_out[:, mask_indices, :] # [B, N_mask, pred_dim]
        
        return predictions_raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference/eval. Returns student representation.
        """
        patches, grid_size = self._process_audio(x)
        x = self.student(patches, grid_size=grid_size)
        return x

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        waveform = batch["waveform"] # [B, 1, T]
        
        patches, current_grid_size = self._process_audio(waveform)
        B, N, D = patches.shape
        
        # Generate shared mask for the batch: [1, N] -> [B, N]
        mask = self.mask_generator(1, device=self.device, grid_size=current_grid_size)
        mask = mask.expand(B, -1)
        
        # Update teacher EMA
        self._update_teacher()
        
        # Compute Student
        student_out = self.compute_student(patches, mask, current_grid_size)
        
        # Compute Predictor
        predictions_raw = self.compute_predictor(student_out, mask, current_grid_size)
        
        # Teacher forward (full)
        with torch.no_grad():
            teacher_full = self.teacher(patches, grid_size=current_grid_size) # [B, N, D]
        
        # Calculate Loss
        loss = self._calculate_jepa_loss(student_out, predictions_raw, teacher_full, mask, current_grid_size)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        waveform = batch["waveform"]
        
        patches, current_grid_size = self._process_audio(waveform)
        B, N, D = patches.shape
        
        # Shared mask for validation as well to enable vectorization
        mask = self.mask_generator(1, device=self.device, grid_size=current_grid_size)
        mask = mask.expand(B, -1)
        
        # Compute Student
        student_out = self.compute_student(patches, mask, current_grid_size)
        
        # Compute Predictor
        predictions_raw = self.compute_predictor(student_out, mask, current_grid_size)
        
        # Teacher forward (full)
        with torch.no_grad():
            teacher_full = self.teacher(patches, grid_size=current_grid_size)
            
            # Calculate Loss
            loss = self._calculate_jepa_loss(student_out, predictions_raw, teacher_full, mask, current_grid_size)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.validation_step(batch, batch_idx)

    def _calculate_jepa_loss(
        self, 
        student_out: torch.Tensor,
        predictions_raw: torch.Tensor, 
        teacher_full: torch.Tensor, 
        mask: torch.Tensor, 
        grid_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Shared JEPA loss calculation logic.
        """
        B = predictions_raw.shape[0]
        m = mask[0]
        mask_indices = torch.nonzero(m).flatten()
        
        # Project back to encoder dimension
        predictions = self.predictor_output_proj(predictions_raw) # [B, N_mask, encoder_dim]
        
        # Targets
        teacher_targets = teacher_full[:, mask_indices, :] # [B, N_mask, encoder_dim]
        
        return self.criterion(predictions, teacher_targets)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.optimizer_config(params=self.parameters())
        
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
