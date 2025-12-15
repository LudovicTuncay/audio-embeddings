import torch
import torchaudio
import numpy as np
from typing import Optional

def resample_and_crop(
    waveform: torch.Tensor, 
    source_sr: int, 
    target_sr: int, 
    max_length: Optional[int] = None
) -> torch.Tensor:
    """
    Resample and optionally crop a waveform.
    
    Args:
        waveform (torch.Tensor): Tensor of shape [T] or [C, T].
        source_sr (int): Source sampling rate.
        target_sr (int): Target sampling rate.
        max_length (Optional[int]): Maximum length in samples (at target_sr).
        
    Returns:
        torch.Tensor: Processed waveform tensor.
    """
    # Resampling and Cropping Logic
    if source_sr != target_sr:
        # We need to resample.
        # Optimization: Crop in source domain first if we have a max_length
        
        if max_length is not None:
            # Calculate required source length to get max_length in target domain
            # Add a small buffer to avoid rounding issues
            crop_len_source = round(max_length * source_sr / target_sr)
            
            if waveform.shape[-1] > crop_len_source:
                max_start = waveform.shape[-1] - crop_len_source
                start = np.random.randint(0, max_start + 1)
                waveform = waveform[..., start : start + crop_len_source]
        
        # Resample
        resampler = torchaudio.transforms.Resample(source_sr, target_sr)
        waveform = resampler(waveform)
        
        # Now handle max_length (trim if we cropped with buffer, or if it was already long enough)
        if max_length is not None and waveform.shape[-1] > max_length:
            waveform = waveform[..., :max_length]
            
    else:
        # No resampling, just standard random crop
        if max_length is not None and waveform.shape[-1] > max_length:
            max_start = waveform.shape[-1] - max_length
            start = np.random.randint(0, max_start + 1)
            waveform = waveform[..., start : start + max_length]
            
    return waveform
