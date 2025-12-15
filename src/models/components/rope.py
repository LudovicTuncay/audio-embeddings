import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim: int, max_res: Tuple[int, int] = (128, 256), temperature: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_h, self.max_w = max_res
        self.temperature = temperature
        
        # Check if dim is divisible by 4 (since we split into 2 for H/W, and each needs 2 for complex)
        assert dim % 4 == 0, "Embedding dimension must be divisible by 4 for 2D RoPE"
        
        dim_h = dim // 2
        dim_w = dim // 2
        
        # Generate frequencies for H and W
        # inv_freq_h: [dim_h // 2]
        inv_freq_h = 1.0 / (temperature ** (torch.arange(0, dim_h, 2).float() / dim_h))
        inv_freq_w = 1.0 / (temperature ** (torch.arange(0, dim_w, 2).float() / dim_w))
        
        self.register_buffer("inv_freq_h", inv_freq_h)
        self.register_buffer("inv_freq_w", inv_freq_w)
        
        # Cache
        self.cached_cos_sin_h = None
        self.cached_cos_sin_w = None

    def _update_cache(self, h: int, w: int, device: torch.device, dtype: torch.dtype):
        # Generate grid
        # We need to support arbitrary positions, but usually we just precompute for max_res
        # or compute on the fly for the given indices.
        # Let's compute for max_res and index into it.
        
        if (self.cached_cos_sin_h is None or self.cached_cos_sin_h[0].shape[0] < h):
            t_h = torch.arange(h, device=device, dtype=dtype)
            freqs_h = torch.einsum("i,j->ij", t_h, self.inv_freq_h) # [H, dim_h/2]
            emb_h = torch.cat((freqs_h, freqs_h), dim=-1) # [H, dim_h]
            self.cached_cos_sin_h = (emb_h.cos(), emb_h.sin())
            
        if (self.cached_cos_sin_w is None or self.cached_cos_sin_w[0].shape[0] < w):
            t_w = torch.arange(w, device=device, dtype=dtype)
            freqs_w = torch.einsum("i,j->ij", t_w, self.inv_freq_w) # [W, dim_w/2]
            emb_w = torch.cat((freqs_w, freqs_w), dim=-1) # [W, dim_w]
            self.cached_cos_sin_w = (emb_w.cos(), emb_w.sin())

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor, grid_size: Tuple[int, int]):
        # q, k: [B, num_heads, N, head_dim]
        # pos_ids: [B, N] or [N] (indices of patches)
        # grid_size: (H, W) - original grid size to decode pos_ids
        
        B, num_heads, N, D = q.shape
        H_grid, W_grid = grid_size
        
        # Decode pos_ids to (h, w)
        # pos_ids are indices in flattened grid [0, H*W-1]
        # h = pos_ids // W_grid
        # w = pos_ids % W_grid
        
        h_idx = pos_ids.div(W_grid, rounding_mode='floor') # [B, N]
        w_idx = pos_ids % W_grid # [B, N]
        
        # Ensure cache is large enough
        self._update_cache(H_grid, W_grid, q.device, q.dtype)
        
        # Fetch cos/sin for H and W
        # cos_h: [B, N, dim_h]
        # We need to gather from cached [max_h, dim_h] using h_idx
        
        # Handle shared pos_ids (if [N])
        if h_idx.ndim == 1:
            h_idx = h_idx.unsqueeze(0).expand(B, -1)
            w_idx = w_idx.unsqueeze(0).expand(B, -1)
            
        cos_h = F.embedding(h_idx, self.cached_cos_sin_h[0]) # [B, N, dim_h]
        sin_h = F.embedding(h_idx, self.cached_cos_sin_h[1])
        cos_w = F.embedding(w_idx, self.cached_cos_sin_w[0]) # [B, N, dim_w]
        sin_w = F.embedding(w_idx, self.cached_cos_sin_w[1])
        
        # Split q, k into halves
        # q: [B, num_heads, N, D] -> [B, N, num_heads, D] for easier manipulation?
        # Usually RoPE is applied on [B, num_heads, N, D] or [N, B, num_heads, D]
        # Let's keep [B, num_heads, N, D]
        
        dim_half = D // 2
        q_h, q_w = q.split(dim_half, dim=-1)
        k_h, k_w = k.split(dim_half, dim=-1)
        
        # Apply RoPE
        # We need to reshape cos/sin to broadcast over num_heads
        # cos_h: [B, N, dim_h] -> [B, 1, N, dim_h]
        cos_h = cos_h.unsqueeze(1)
        sin_h = sin_h.unsqueeze(1)
        cos_w = cos_w.unsqueeze(1)
        sin_w = sin_w.unsqueeze(1)
        
        q_h_rot = self._apply_rotary(q_h, cos_h, sin_h)
        k_h_rot = self._apply_rotary(k_h, cos_h, sin_h)
        
        q_w_rot = self._apply_rotary(q_w, cos_w, sin_w)
        k_w_rot = self._apply_rotary(k_w, cos_w, sin_w)
        
        q_rot = torch.cat((q_h_rot, q_w_rot), dim=-1)
        k_rot = torch.cat((k_h_rot, k_w_rot), dim=-1)
        
        return q_rot, k_rot

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [B, num_heads, N, dim_half]
        # cos, sin: [B, 1, N, dim_half]
        # Standard RoPE rotation:
        # x = [x1, x2]
        # out = [x1*cos - x2*sin, x1*sin + x2*cos]
        # This assumes pairs are adjacent.
        # My inv_freq generation: cat(freqs, freqs).
        # This corresponds to x = [x_first_half, x_second_half] pairing?
        # Usually RoPE pairs even/odd or first/second half.
        # "The standard implementation ... pairs feature i with i + d/2"
        # My emb generation: cat(freqs, freqs) -> [f0, f1, ..., f0, f1, ...] ? No.
        # freqs is [0, 2, ...]
        # cat(freqs, freqs) -> [f0, f2, ..., f0, f2, ...]
        # So it expects x to be split into two halves and rotated.
        # rotate_half(x) = [-x2, x1]
        
        return (x * cos) + (self._rotate_half(x) * sin)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope: Optional[RotaryEmbedding2D] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.rope = rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos_ids: torch.Tensor = None, grid_size: Tuple[int, int] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, num_heads, N, head_dim]

        if self.rope is not None and pos_ids is not None and grid_size is not None:
            q, k = self.rope(q, k, pos_ids, grid_size)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
