import torch
import hydra
from omegaconf import OmegaConf
import sys
import os

# Add src to python path
sys.path.append(os.getcwd())

from src.models.rq_jepa_module import RQJEPAModule

def verify_rq_jepa():
    print("Verifying RQ-JEPA Implementation...")

    # Mock Net Config
    net_config = {
        "spectrogram": {},
        "patch_embed": {"img_size": (1024, 128), "patch_size": (16, 16), "in_chans": 1, "embed_dim": 768},
        "masking": {"input_size": (64, 8), "mask_ratio": (0.4, 0.6)}, # approx grid size for 1024x128
        "encoder": {"img_size": (1024, 128), "patch_size": (16, 16), "embed_dim": 768, "depth": 2, "num_heads": 4},
        "predictor": {"img_size": (1024, 128), "patch_size": (16, 16), "embed_dim": 384, "depth": 1, "num_heads": 4}
    }

    # Instantiate Module
    model = RQJEPAModule(
        optimizer=lambda params: torch.optim.Adam(params),
        net=net_config,
        rq_lambda=0.5,
        codebook_dim=16,
        vocab_size=100, # Small vocab for test
        jepa_criterion=torch.nn.MSELoss(),
        rq_criterion=torch.nn.CrossEntropyLoss()
    )
    
    # Mock Batch
    B = 2
    T = 16000 * 2 # 2 seconds
    waveform = torch.randn(B, 1, T)
    batch = {"waveform": waveform}
    
    print("Model instantiated. Running training_step...")
    
    # Run training step
    loss = model.training_step(batch, batch_idx=0)
    
    print(f"Training step successful. Loss: {loss.item()}")
    
    # Check if sub-losses are computed (by inspecting if code ran without error and returned scalar)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    
    print("Verification Passed!")

if __name__ == "__main__":
    verify_rq_jepa()
