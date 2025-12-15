import torch
import torch.nn as nn
import torchaudio
from typing import Optional

class Spectrogram(nn.Module):
    """
    Mel Spectrogram module with AmplitudeToDB conversion.
    
    Args:
        sample_rate (int): Sample rate of the audio.
        n_fft (int): Size of FFT.
        win_length (Optional[int]): Window length. Defaults to n_fft.
        win_length_ms (Optional[float]): Window length in milliseconds. Overrides win_length if provided.
        hop_length (Optional[int]): Hop length. Defaults to win_length // 2.
        hop_length_ms (Optional[float]): Hop length in milliseconds. Overrides hop_length if provided.
        n_mels (int): Number of mel filterbanks.
        f_min (float): Minimum frequency.
        f_max (Optional[float]): Maximum frequency.
        power (float): Power of the magnitude.
    """
    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 4096,
        win_length: Optional[int] = None,
        win_length_ms: Optional[float] = None,
        hop_length: Optional[int] = None,
        hop_length_ms: Optional[float] = None,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        power: float = 2.0,
    ):
        super().__init__()
        
        if win_length is None:
            if win_length_ms is None:
                win_length = n_fft
            else:
                win_length = int(sample_rate * win_length_ms / 1000)
                
        if hop_length is None:
            if hop_length_ms is None:
                hop_length = win_length // 2
            else:
                hop_length = int(sample_rate * hop_length_ms / 1000)

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=power,
            normalized=True
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input waveform [B, C, T] or [B, T].
            
        Returns:
            torch.Tensor: Log-Mel Spectrogram [B, C, F, T].
        """
        # x: [B, C, T]
        # MelSpectrogram expects [..., T]
        # Output will be [..., n_mels, time]
        
        spec = self.mel_spec(x)
        spec = self.amplitude_to_db(spec)
        
        return spec
