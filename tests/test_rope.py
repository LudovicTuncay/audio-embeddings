import pytest
import torch

from src.models.components.rope import RotaryEmbedding2D


def test_rotary_embedding_requires_dim_divisible_by_four() -> None:
    with pytest.raises(AssertionError, match="divisible by 4"):
        RotaryEmbedding2D(dim=10)


def test_rotary_embedding_preserves_tensor_shapes() -> None:
    rope = RotaryEmbedding2D(dim=8, max_res=(2, 2))

    q = torch.randn(2, 1, 4, 8)
    k = torch.randn(2, 1, 4, 8)
    pos_ids = torch.arange(4)

    q_rot, k_rot = rope(q, k, pos_ids=pos_ids, grid_size=(2, 2))

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rotary_embedding_axis_selectivity() -> None:
    rope = RotaryEmbedding2D(dim=64, max_res=(4, 4))
    q = torch.ones(1, 1, 16, 64)
    k = torch.ones(1, 1, 16, 64)
    pos_ids = torch.arange(16).unsqueeze(0)

    q_rot, _ = rope(q, k, pos_ids=pos_ids, grid_size=(4, 4))
    q_grid = q_rot.reshape(4, 4, 64)

    diff_w_first_half = (q_grid[0, 0, :32] - q_grid[0, 1, :32]).abs().sum()
    diff_w_second_half = (q_grid[0, 0, 32:] - q_grid[0, 1, 32:]).abs().sum()
    diff_h_first_half = (q_grid[0, 0, :32] - q_grid[1, 0, :32]).abs().sum()
    diff_h_second_half = (q_grid[0, 0, 32:] - q_grid[1, 0, 32:]).abs().sum()

    assert diff_w_first_half < 1e-5
    assert diff_w_second_half > 1.0
    assert diff_h_first_half > 1.0
    assert diff_h_second_half < 1e-5
