import torch

from src.models.components.masking import MaskingGenerator


def test_masking_generator_shape_dtype_and_device() -> None:
    generator = MaskingGenerator(
        input_size=(16, 16),
        patch_size=(4, 4),
        mask_ratio=(0.25, 0.25),
    )

    mask = generator(batch_size=3, device=torch.device("cpu"))

    assert mask.shape == (3, 16)
    assert mask.dtype == torch.bool
    assert mask.device.type == "cpu"
    assert mask.sum(dim=1).tolist() == [4, 4, 4]


def test_masking_generator_grid_size_override() -> None:
    generator = MaskingGenerator(
        input_size=(16, 16),
        patch_size=(4, 4),
        mask_ratio=(0.5, 0.5),
    )

    mask = generator(batch_size=2, grid_size=(2, 3))

    assert mask.shape == (2, 6)
    assert mask.sum(dim=1).tolist() == [3, 3]
