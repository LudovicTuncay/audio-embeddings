import torch
import numpy as np
import torchaudio
from src.data.audioset_datamodule import AudioSetDataset

# Mock Dataset to test resampling logic
class MockAudioSetDatasetResample(AudioSetDataset):
    def __init__(self, source_sr, target_sr, max_length=None):
        self.source_sample_rate = source_sr
        self.target_sample_rate = target_sr
        self.max_length = max_length
        self.transform = None
        self.valid_indices = [0]
        self.h5_file = None
        
    def _open_h5(self):
        pass
        
    def __getitem__(self, idx):
        # Create a 10s sine wave at source SR
        duration = 10
        t = np.linspace(0, duration, int(self.source_sample_rate * duration))
        waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Mock loading
        waveform = torch.from_numpy(waveform)
        
        # --- Copy-paste logic from AudioSetDataset.__getitem__ ---
        # Resampling and Cropping Logic
        if self.source_sample_rate != self.target_sample_rate:
            if self.max_length is not None:
                crop_len_source = int(self.max_length * self.source_sample_rate / self.target_sample_rate) + 100
                if len(waveform) > crop_len_source:
                    max_start = len(waveform) - crop_len_source
                    start = np.random.randint(0, max_start + 1)
                    waveform = waveform[start : start + crop_len_source]
            
            resampler = torchaudio.transforms.Resample(self.source_sample_rate, self.target_sample_rate)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
            
            if self.max_length is not None and len(waveform) > self.max_length:
                waveform = waveform[:self.max_length]
        else:
            if self.max_length is not None and len(waveform) > self.max_length:
                max_start = len(waveform) - self.max_length
                start = np.random.randint(0, max_start + 1)
                waveform = waveform[start : start + self.max_length]
        # ---------------------------------------------------------
        
        return waveform

def test_resampling():
    source_sr = 32000
    target_sr = 16000
    max_len_target = 160000 # 10s @ 16kHz
    
    dataset = MockAudioSetDatasetResample(source_sr, target_sr, max_length=max_len_target)
    
    print(f"Testing resampling from {source_sr} to {target_sr} with max_len {max_len_target}")
    
    waveform = dataset[0]
    print(f"Output shape: {waveform.shape}")
    
    if waveform.shape[0] == max_len_target:
        print("PASS: Output length matches max_length")
    else:
        print(f"FAIL: Output length {waveform.shape[0]} != {max_len_target}")
        
    # Test without max_length
    dataset_no_max = MockAudioSetDatasetResample(source_sr, target_sr, max_length=None)
    waveform_full = dataset_no_max[0]
    expected_len = 160000 # 10s * 16000
    print(f"Output shape (no max): {waveform_full.shape}")
    if abs(waveform_full.shape[0] - expected_len) < 100:
        print("PASS: Output length matches expected resampled length")
    else:
        print(f"FAIL: Output length {waveform_full.shape[0]} != {expected_len}")

if __name__ == "__main__":
    test_resampling()
