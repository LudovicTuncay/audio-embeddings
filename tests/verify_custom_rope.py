import sys
import os
import torch
sys.path.append(os.path.abspath("."))
from src.models.components.rope import RotaryEmbedding2D

def verify_custom_rope():
    dim = 64
    rope = RotaryEmbedding2D(dim, max_res=(4, 4))
    
    # Input: [B, num_heads, N, D]
    # B=1, num_heads=1, N=16, D=64
    # B=1, num_heads=1, N=16, D=64
    # Use constant input to verify RoPE effect only
    q = torch.ones(1, 1, 16, 64)
    k = torch.ones(1, 1, 16, 64)
    
    # pos_ids for 4x4 grid
    pos_ids = torch.arange(16).unsqueeze(0) # [1, 16]
    grid_size = (4, 4)
    
    q_rot, k_rot = rope(q, k, pos_ids, grid_size)
    
    print(f"Output shape: {q_rot.shape}")
    
    # Reshape to grid [H, W, D]
    q_grid = q_rot.reshape(4, 4, 64)
    
    # Check diff along W (0,0) vs (0,1)
    # Should ONLY affect the second half (W part)
    # First half (H part) should be IDENTICAL because H is same (0)
    
    diff_w_first_half = (q_grid[0, 0, :32] - q_grid[0, 1, :32]).abs().sum()
    diff_w_second_half = (q_grid[0, 0, 32:] - q_grid[0, 1, 32:]).abs().sum()
    
    print(f"Diff W (First Half - H part): {diff_w_first_half}")
    print(f"Diff W (Second Half - W part): {diff_w_second_half}")
    
    # Check diff along H (0,0) vs (1,0)
    # Should ONLY affect the first half (H part)
    # Second half (W part) should be IDENTICAL because W is same (0)
    
    diff_h_first_half = (q_grid[0, 0, :32] - q_grid[1, 0, :32]).abs().sum()
    diff_h_second_half = (q_grid[0, 0, 32:] - q_grid[1, 0, 32:]).abs().sum()
    
    print(f"Diff H (First Half - H part): {diff_h_first_half}")
    print(f"Diff H (Second Half - W part): {diff_h_second_half}")
    
    # Assertions
    assert diff_w_first_half < 1e-5, "First half should not change with W"
    assert diff_w_second_half > 1.0, "Second half should change with W"
    assert diff_h_first_half > 1.0, "First half should change with H"
    assert diff_h_second_half < 1e-5, "Second half should not change with H"
    
    print("Verification Successful!")

if __name__ == "__main__":
    verify_custom_rope()
