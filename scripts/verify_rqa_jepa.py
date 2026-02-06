import torch
import torch.nn as nn
import sys
import os

# Add src to python path
sys.path.append(os.getcwd())

from src.models.rqa_jepa_module import RQAJEPAModule


def verify_rqa_jepa():
    print("Verifying RQA-JEPA Implementation...")

    # Mock Net Config
    net_config = {
        "spectrogram": {},
        "patch_embed": {
            "img_size": (1024, 128),
            "patch_size": (16, 16),
            "in_chans": 1,
            "embed_dim": 768,
        },
        "masking": {
            "input_size": (64, 8),
            "mask_ratio": (0.4, 0.6),
        },  # approx grid size for 1024x128
        "encoder": {
            "img_size": (1024, 128),
            "patch_size": (16, 16),
            "embed_dim": 768,
            "depth": 2,
            "num_heads": 4,
        },
        "predictor": {
            "img_size": (1024, 128),
            "patch_size": (16, 16),
            "embed_dim": 384,
            "depth": 1,
            "num_heads": 4,
        },
    }

    # Mock Optimizer
    def optimizer_partial(params):
        return torch.optim.Adam(params)

    # --- Mode 1: Teacher Input ---
    print("\n--- Verifying RQA-JEPA (Mode: teacher) ---")
    model_teacher = RQAJEPAModule(
        optimizer=optimizer_partial,
        net=net_config,
        jepa_criterion=nn.MSELoss(),
        rq_criterion=nn.CrossEntropyLoss(),
        rq_input_type="teacher",
        codebook_dim=16,
        vocab_size=100,
    )

    # Mock input data
    B, C, T = 2, 1, 16000  # 1 second audio
    waveform = torch.rand(B, C, T)
    batch = {"waveform": waveform}

    # Set device
    model_teacher.to("cpu")

    # Run training_step
    print("Model (teacher) instantiated. Running training_step...")
    loss_teacher = model_teacher.training_step(batch, 0)
    print(f"Training step successful. Loss: {loss_teacher.item()}")
    assert isinstance(loss_teacher, torch.Tensor)
    assert loss_teacher.ndim == 0

    # --- Mode 2: Spectrogram Input ---
    print("\n--- Verifying RQA-JEPA (Mode: spectrogram) ---")
    model_spec = RQAJEPAModule(
        optimizer=optimizer_partial,
        net=net_config,
        jepa_criterion=nn.MSELoss(),
        rq_criterion=nn.CrossEntropyLoss(),
        rq_input_type="spectrogram",
        codebook_dim=16,
        vocab_size=100,
    )
    model_spec.to("cpu")

    print("Model (spectrogram) instantiated. Running training_step...")
    loss_spec = model_spec.training_step(batch, 0)
    print(f"Training step successful. Loss: {loss_spec.item()}")
    assert isinstance(loss_spec, torch.Tensor)
    assert loss_spec.ndim == 0

    print("\nVerification Passed!")


if __name__ == "__main__":
    verify_rqa_jepa()
