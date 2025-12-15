import torch
import numpy as np
from typing import Tuple, Optional

class MaskingGenerator:
    """
    Generates masks for the input patches.
    
    Args:
        input_size (tuple[int, int]): Input image size (H, W).
        patch_size (tuple[int, int]): Patch size (H, W).
        mask_ratio (tuple[float, float]): Range of mask ratio (min, max).
    """
    def __init__(
        self,
        input_size: Tuple[int, int] = (128, 256),
        patch_size: Tuple[int, int] = (16, 16),
        mask_ratio: Tuple[float, float] = (0.4, 0.6),
    ):
        self.height, self.width = input_size
        self.patch_h, self.patch_w = patch_size
        self.num_patches_h = self.height // self.patch_h
        self.num_patches_w = self.width // self.patch_w
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.mask_ratio = mask_ratio

    def __call__(
        self, 
        batch_size: int, 
        device: torch.device = torch.device('cpu'), 
        grid_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Generate masks for a batch.
        
        Args:
            batch_size (int): Batch size.
            device (torch.device): Device to place the masks on.
            grid_size (Optional[Tuple[int, int]]): Grid size (H, W) if different from init.
            
        Returns:
            torch.Tensor: Masks [B, N] (boolean, True=masked).
        """
        masks = []
        for _ in range(batch_size):
            mask = self._generate_mask(grid_size)
            masks.append(mask)
        
        return torch.stack(masks).to(device)

    def _generate_mask(self, grid_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Generate a single mask.
        """
        if grid_size is not None:
            num_patches_h, num_patches_w = grid_size
            num_patches = num_patches_h * num_patches_w
        else:
            num_patches = self.num_patches
            
        mask = torch.zeros(num_patches, dtype=torch.bool)
        
        target_masked = int(num_patches * np.random.uniform(*self.mask_ratio))
        
        # Random Permutation Masking
        if target_masked > 0:
            perm = torch.randperm(num_patches)
            mask_indices = perm[:target_masked]
            mask[mask_indices] = True
            
        return mask # Already flattened
