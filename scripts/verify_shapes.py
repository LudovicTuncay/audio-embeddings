import torch
import torchaudio
import math

def get_spectrogram_shape(waveform_len, hop_length=1250, center=True):
    # Simulation of torchaudio MelSpectrogram shape calculation
    # If center=True, it pads the signal.
    # Output time steps = input_samples // hop_length + 1
    return waveform_len // hop_length + 1

def calculate_required_length(current_len, hop_length, patch_time_dim):
    # We need spec_len % (2 * patch_time_dim) == 0
    # spec_len = current_len // hop_length + 1
    
    # Let target_spec_len be the next multiple of (2 * patch_time_dim)
    # target_spec_len = ceil(spec_len / (2 * patch_time_dim)) * (2 * patch_time_dim)
    
    # Then we need waveform_len such that waveform_len // hop_length + 1 = target_spec_len
    # waveform_len // hop_length = target_spec_len - 1
    # waveform_len = (target_spec_len - 1) * hop_length
    # But wait, we can just pad the waveform to be larger.
    # Any waveform_len in range [ (target_spec_len-1)*hop_length, target_spec_len*hop_length - 1 ] might work?
    # Let's just pick one: waveform_len = (target_spec_len - 1) * hop_length
    
    spec_len = current_len // hop_length + 1
    block_size = 2 * patch_time_dim
    
    if spec_len % block_size == 0:
        target_spec_len = spec_len
    else:
        target_spec_len = (spec_len // block_size + 1) * block_size
        
    # Reverse to waveform length
    # We want (target_wave_len // hop_length + 1) == target_spec_len
    # target_wave_len // hop_length = target_spec_len - 1
    # target_wave_len = (target_spec_len - 1) * hop_length
    
    return target_spec_len, target_spec_len * hop_length # Approximate, let's verify

def test_shapes():
    hop_length = 1250
    patch_time_dim = 16
    
    lengths = [32000, 48000, 320000, 12345]
    
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=32000,
        n_fft=4096,
        win_length=4096,
        hop_length=hop_length,
        n_mels=128,
        center=True
    )
    
    print(f"Testing with hop_length={hop_length}, patch_time_dim={patch_time_dim}")
    
    for l in lengths:
        wave = torch.randn(1, l)
        spec = mel(wave)
        spec_len = spec.shape[-1]
        print(f"Wave: {l}, Spec: {spec_len}")
        
        # Calculate required
        target_spec_len, target_wave_len = calculate_required_length(l, hop_length, patch_time_dim)
        
        # Verify
        wave_pad = torch.randn(1, target_wave_len)
        spec_pad = mel(wave_pad)
        spec_pad_len = spec_pad.shape[-1]
        
        print(f"  Target Spec: {target_spec_len}, Target Wave: {target_wave_len}")
        print(f"  Actual Spec: {spec_pad_len}")
        print(f"  Even patches? {spec_pad_len / patch_time_dim} (Time patches)")
        print(f"  Even time patches condition: {(spec_pad_len // patch_time_dim) % 2 == 0}")
        
        if spec_pad_len != target_spec_len:
            print("  MISMATCH!")

if __name__ == "__main__":
    test_shapes()
