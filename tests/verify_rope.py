import torch
from timm.layers import RotaryEmbedding

def verify_rope():
    # 2D Grid: H=4, W=4
    # Dim=64 (Head dim)
    # We want half dim for H, half for W? Or how does timm handle it?
    
    dim = 64
    rope = RotaryEmbedding(dim, feat_shape=[4, 4])
    
    # Input: [B, H, N, D] -> [1, 1, 16, 64]
    x = torch.randn(1, 1, 16, 64)
    
    # Forward
    x_rope = rope(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_rope.shape}")
    
    # Check if it varies along H and W
    # Reshape to [H, W, D]
    x_grid = x_rope.reshape(4, 4, 64)
    
    # Check difference between (0,0) and (0,1) -> W change
    diff_w = (x_grid[0, 0] - x_grid[0, 1]).abs().sum()
    print(f"Diff along W: {diff_w}")
    
    # Check difference between (0,0) and (1,0) -> H change
    diff_h = (x_grid[0, 0] - x_grid[1, 0]).abs().sum()
    print(f"Diff along H: {diff_h}")
    
    # If it's 1D RoPE on flattened sequence, diff_w and diff_h would both be non-zero but structure might be different.
    # If it's 2D, it should encode H and W separately.
    
    # Let's check if the embedding is indeed 2D.
    # Usually 2D RoPE splits D into D/2 for H and D/2 for W.
    # Let's see if the first half changes with H and second half with W?
    
    # Change in W (0,0) vs (0,1)
    # Should affect one half?
    diff_w_first_half = (x_grid[0, 0, :32] - x_grid[0, 1, :32]).abs().sum()
    diff_w_second_half = (x_grid[0, 0, 32:] - x_grid[0, 1, 32:]).abs().sum()
    
    print(f"Diff W (First Half): {diff_w_first_half}")
    print(f"Diff W (Second Half): {diff_w_second_half}")
    
    # Change in H (0,0) vs (1,0)
    diff_h_first_half = (x_grid[0, 0, :32] - x_grid[1, 0, :32]).abs().sum()
    diff_h_second_half = (x_grid[0, 0, 32:] - x_grid[1, 0, 32:]).abs().sum()
    
    print(f"Diff H (First Half): {diff_h_first_half}")
    print(f"Diff H (Second Half): {diff_h_second_half}")

if __name__ == "__main__":
    verify_rope()
