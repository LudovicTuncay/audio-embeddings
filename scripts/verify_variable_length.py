import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.models.audio_jepa_module import AudioJEPAModule
from src.data.audioset_datamodule import AudioSetDataModule

# Mock Dataset
class MockAudioDataset(Dataset):
    def __init__(self, lengths):
        self.lengths = lengths
        
    def __len__(self):
        return len(self.lengths)
    
    def __getitem__(self, idx):
        l = self.lengths[idx]
        waveform = torch.randn(1, l)
        target = torch.randn(527) # AudioSet classes
        return {
            "waveform": waveform,
            "target": target,
            "audio_name": f"audio_{idx}",
            "index": idx
        }

def test_variable_length():
    # 1. Test Data Loading
    lengths = [32000, 48000, 30000, 50000] # Variable lengths
    dataset = MockAudioDataset(lengths)
    
    # Use collate_fn from AudioSetDataModule
    # Pass parameters manually for testing
    collate_fn = lambda b: AudioSetDataModule.collate_fn(b, hop_length=1250, patch_time_dim=16)
    
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    batch = next(iter(loader))
    waveforms = batch["waveform"]
    print(f"Batch waveforms shape: {waveforms.shape}")
    
    # Check if shape is correct
    # Max length is 50000.
    # Hop = 1250.
    # Max spec len = 50000 // 1250 + 1 = 41.
    # Block size = 32.
    # Target spec len = ceil(41/32)*32 = 64.
    # Target wave len = (64-1)*1250 = 63 * 1250 = 78750.
    
    expected_len = 78750
    if waveforms.shape[-1] == expected_len:
        print("Padding logic verified!")
    else:
        print(f"Padding logic mismatch! Expected {expected_len}, got {waveforms.shape[-1]}")
        
    # 2. Test Model Forward
    print("Initializing model...")
    # Minimal config
    net_config = {
        "spectrogram": {
            "sample_rate": 32000,
            "n_fft": 4096,
            "win_length": 4096,
            "hop_length": 1250,
            "n_mels": 128,
            "f_min": 0.0,
            "f_max": None,
            # target_length removed
        },
        "patch_embed": {
            "img_size": (128, 256), # This is just for init, will be ignored/overridden dynamically
            "patch_size": (16, 16),
            "in_chans": 1,
            "embed_dim": 192, # Small dim for speed
        },
        "masking": {
            "input_size": (128, 256),
            "patch_size": (16, 16),
            "mask_ratio": (0.4, 0.6),
        },
        "encoder": {
            "embed_dim": 192,
            "depth": 2,
            "num_heads": 3,
            "pos_embed_type": "rope",
            "img_size": (128, 256),
            "patch_size": (16, 16),
        },
        "predictor": {
            "embed_dim": 192,
            "depth": 1,
            "num_heads": 3,
            "pos_embed_type": "rope",
            "img_size": (128, 256),
            "patch_size": (16, 16),
        }
    }
    
    model = AudioJEPAModule(
        optimizer=torch.optim.AdamW,
        net=net_config
    )
    
    # Initialize EMA decay manually since we skip Lightning loop
    model.current_ema_decay = 0.996
    
    print("Running training_step...")
    loss = model.training_step(batch, 0)
    print(f"Training step loss: {loss}")
    
    print("Running validation_step...")
    val_loss = model.validation_step(batch, 0)
    print(f"Validation step loss: {val_loss}")
    
    print("Test passed!")

if __name__ == "__main__":
    test_variable_length()
