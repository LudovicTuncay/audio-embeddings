import torch
import torch.nn as nn
from timm.layers import Mlp, build_sincos2d_pos_embed, DropPath
from src.models.components.rope import RoPEAttention, RotaryEmbedding2D
from typing import Optional, Tuple, Union

class RoPEBlock(nn.Module):
    """
    Transformer Block with RoPE support.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        rope: Optional[RotaryEmbedding2D] = None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            rope=rope,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, pos_ids: Optional[torch.Tensor] = None, grid_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), pos_ids=pos_ids, grid_size=grid_size))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ViT(nn.Module):
    """
    Vision Transformer with support for RoPE and 2D positional embeddings.
    
    Args:
        embed_dim (int): Embedding dimension.
        depth (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        qkv_bias (bool): Enable bias for QKV projections.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        drop_path_rate (float): Stochastic depth rate.
        norm_layer (nn.Module): Normalization layer.
        act_layer (nn.Module): Activation layer.
        num_patches (int): Total number of patches (used for learnable/sincos pos embed).
        img_size (tuple[int, int]): Input image size (H, W).
        patch_size (tuple[int, int]): Patch size (H, W).
        pos_embed_type (str): Type of positional embedding ("rope", "sincos", "learnable").
    """
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        num_patches: int = 128,
        img_size: tuple[int, int] = (128, 256),
        patch_size: tuple[int, int] = (16, 16),
        pos_embed_type: str = "rope",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.pos_embed_type = pos_embed_type
        
        # Positional Embeddings
        if pos_embed_type == "rope":
            head_dim = embed_dim // num_heads
            self.rope = RotaryEmbedding2D(dim=head_dim, max_res=self.grid_size)
            self.pos_embed = None
        elif pos_embed_type == "sincos":
            self.rope = None
            # build_sincos2d_pos_embed(feat_shape, dim, ...)
            # We assume grid_size matches num_patches
            pos_embed = build_sincos2d_pos_embed(self.grid_size, embed_dim)
            self.register_buffer('pos_embed', pos_embed.unsqueeze(0)) # [1, N, D]
        elif pos_embed_type == "learnable":
            self.rope = None
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")
        
        # Stochastic Depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            RoPEBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                rope=self.rope,
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self, 
        x: torch.Tensor, 
        pos_ids: Optional[torch.Tensor] = None, 
        add_pos_embed: bool = True, 
        grid_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [B, N, D].
            pos_ids (Optional[torch.Tensor]): Positional indices [B, N] or [N].
            add_pos_embed (bool): Whether to add positional embeddings (for non-RoPE).
            grid_size (Optional[Tuple[int, int]]): Grid size for RoPE/PosEmbed.
            
        Returns:
            torch.Tensor: Output tensor [B, N, D].
        """
        # Determine grid size
        if grid_size is None:
            if pos_ids is None:
                # Infer from x assuming full sequence
                B, N, D = x.shape
                H_grid = self.grid_size[0]
                W_grid = N // H_grid
                current_grid_size = (H_grid, W_grid)
            else:
                # Cannot infer, use default (might be wrong if variable length)
                current_grid_size = self.grid_size
        else:
            current_grid_size = grid_size

        if self.pos_embed_type != "rope" and add_pos_embed:
            if pos_ids is not None:
                # Select positional embeddings
                if pos_ids.ndim == 1:
                    # Shared pos_ids across batch
                    pos_embed = self.pos_embed[:, pos_ids, :] # [1, N_subset, D]
                else:
                    # Different pos_ids per sample
                    pos_embed = self.pos_embed.expand(x.shape[0], -1, -1)
                    pos_embed = torch.gather(
                        pos_embed, 
                        1, 
                        pos_ids.unsqueeze(-1).expand(-1, -1, self.embed_dim)
                    )
                x = x + pos_embed
            else:
                # Assume full sequence
                if x.shape[1] == self.num_patches:
                    x = x + self.pos_embed
                elif self.pos_embed is not None and x.shape[1] <= self.pos_embed.shape[1]:
                     x = x + self.pos_embed[:, :x.shape[1], :]

        # For RoPE, we need pos_ids. If not provided, generate them.
        if self.pos_embed_type == "rope" and pos_ids is None:
            device = x.device
            # We need to generate pos_ids for the current grid
            # If we inferred current_grid_size, we should use it.
            # pos_ids should be 0..N-1
            B, N, D = x.shape
            pos_ids = torch.arange(N, device=device)

        for block in self.blocks:
            x = block(x, pos_ids=pos_ids, grid_size=current_grid_size)
            
        x = self.norm(x)
        return x
