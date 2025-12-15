import torch
import torch.nn as nn
from timm.layers import PatchEmbed as TimmPatchEmbed

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding.
    
    Args:
        img_size (tuple[int, int]): Input image size (H, W).
        patch_size (tuple[int, int]): Patch size (H, W).
        in_chans (int): Number of input channels.
        embed_dim (int): Embedding dimension.
    """
    def __init__(
        self,
        img_size: tuple[int, int] = (128, 256),
        patch_size: tuple[int, int] = (16, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_embed = TimmPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten=True,
            bias=True,
            strict_img_size=False,
        )
        self.num_patches = self.patch_embed.num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W].
            
        Returns:
            torch.Tensor: Patch embeddings [B, N, D].
        """
        return self.patch_embed(x)
