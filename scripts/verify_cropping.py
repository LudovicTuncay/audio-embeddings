import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.data.audioset_datamodule import AudioSetDataset

# Mock Dataset inheriting from AudioSetDataset to test logic without H5
class MockAudioSetDataset(AudioSetDataset):
    def __init__(self, lengths, max_length=None):
        self.lengths = lengths
        self.max_length = max_length
        self.transform = None
        self.valid_indices = list(range(len(lengths)))
        self.h5_file = None # Not used
        
    def _open_h5(self):
        pass
        
    def __getitem__(self, idx):
        # Mock waveform loading
        l = self.lengths[idx]
        # Create a waveform where values are 0..L-1 so we can check cropping start
        waveform = np.arange(l, dtype=np.float32)
        
        # Random Crop logic from AudioSetDataset
        if self.max_length is not None and len(waveform) > self.max_length:
            max_start = len(waveform) - self.max_length
            start = np.random.randint(0, max_start + 1)
            waveform = waveform[start : start + self.max_length]
            
        # Mock other returns
        target = torch.zeros(527)
        audio_name = f"audio_{idx}"
        
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        
        return {
            "waveform": waveform,
            "target": target,
            "audio_name": audio_name,
            "index": idx
        }

def test_random_cropping():
    max_len = 100
    lengths = [50, 100, 150, 200]
    
    dataset = MockAudioSetDataset(lengths, max_length=max_len)
    
    print(f"Testing with max_length={max_len}")
    
    for i in range(len(lengths)):
        # Test multiple times to check randomness
        starts = []
        for _ in range(5):
            item = dataset[i]
            wave = item["waveform"]
            # Check length
            if wave.shape[-1] > max_len:
                print(f"FAIL: Index {i} (orig {lengths[i]}) has length {wave.shape[-1]} > {max_len}")
            
            # Check content (start index)
            start_val = wave[0, 0].item()
            starts.append(start_val)
            
        print(f"Index {i} (orig {lengths[i]}): Starts = {starts}")
        
        if lengths[i] > max_len:
            # Should be cropped to max_len
            if wave.shape[-1] != max_len:
                 print(f"FAIL: Index {i} should be cropped to {max_len}, got {wave.shape[-1]}")
            
            # Should be random (unless max_start=0)
            if len(set(starts)) == 1 and lengths[i] > max_len + 5: # Allow some chance of collision
                 print(f"WARNING: Index {i} might not be random? Starts: {starts}")
        else:
            # Should be original length
            if wave.shape[-1] != lengths[i]:
                print(f"FAIL: Index {i} should be {lengths[i]}, got {wave.shape[-1]}")
                
    print("Test finished.")

if __name__ == "__main__":
    test_random_cropping()
