import sys
import os
import torch

# Add src to path
sys.path.append(os.path.abspath("src"))

from data.audioset_datamodule import AudioSetDataModule

def verify_data():
    print("Initializing DataModule...")
    dm = AudioSetDataModule(
        data_dir="data/AudioSet",
        batch_size=4,
        num_workers=0 # Use 0 for debugging
    )
    dm.setup()
    
    print(f"Train dataset size: {len(dm.train_dataset)}")
    print(f"Val dataset size: {len(dm.val_dataset)}")
    
    print("Fetching a batch...")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    
    waveform = batch['waveform']
    target = batch['target']
    audio_name = batch['audio_name']
    index = batch['index']
    
    print(f"Waveform shape: {waveform.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Audio names: {audio_name}")
    print(f"Indices: {index}")
    
    assert waveform.ndim == 3, "Waveform should be [B, C, T]"
    assert waveform.shape[1] == 1, "Channel dim should be 1"
    assert waveform.shape[2] == 320000, "Time dim should be 320000"
    
    print("Verification successful!")

if __name__ == "__main__":
    verify_data()
