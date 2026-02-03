import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import numpy as np
from typing import Optional, List, Dict, Any, Union
from functools import partial

class MockAudioSetDataset(Dataset):
    """
    Mock Dataset for AudioSet data that generates random noise.
    """
    def __init__(
        self, 
        length: int = 100,
        max_length: int = 160000,
        target_sample_rate: int = 16000
    ):
        self.length = length
        self.max_length = max_length
        self.target_sample_rate = target_sample_rate
        
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        # Generate random waveform [1, T]
        # Random length between max_length // 2 and max_length for realism, or just fixed max_length
        # Let's do fixed max_length for simplicity in mock
        waveform = torch.randn(1, self.max_length)
        
        # Fake target (multi-hot) - assuming AudioSet has 527 classes
        target = torch.zeros(527)
        # Set a few random classes to 1
        indices = torch.randint(0, 527, (3,))
        target[indices] = 1.0
        
        audio_name = f"mock_audio_{idx}"
            
        return {
            "waveform": waveform,
            "target": target,
            "audio_name": audio_name,
            "index": idx
        }

class MockAudioSetDataModule(L.LightningDataModule):
    """
    LightningDataModule for Mock AudioSet.
    """
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = True,
        max_audio_length_sec: float = 10.0,
        target_sample_rate: int = 16000,
        collate_mode: str = "pad",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_mode = collate_mode
        
        self.max_audio_length = int(max_audio_length_sec * target_sample_rate)
        
        self.train_dataset: Optional[MockAudioSetDataset] = None
        self.val_dataset: Optional[MockAudioSetDataset] = None
        self.test_dataset: Optional[MockAudioSetDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = MockAudioSetDataset(
                length=1000, # Fake dataset size
                max_length=self.max_audio_length,
                target_sample_rate=self.hparams.target_sample_rate
            )
            self.val_dataset = MockAudioSetDataset(
                length=100,
                max_length=self.max_audio_length,
                target_sample_rate=self.hparams.target_sample_rate
            )

        if stage == "test":
            self.test_dataset = MockAudioSetDataset(
                length=50,
                max_length=self.max_audio_length,
                target_sample_rate=self.hparams.target_sample_rate
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=partial(self.collate_fn, mode=self.collate_mode)
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=partial(self.collate_fn, mode=self.collate_mode)
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=partial(self.collate_fn, mode=self.collate_mode)
        )

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]], mode: str = "pad") -> Dict[str, Any]:
        """
        Collate function to pad or truncate waveforms.
        """
        waveforms = [item['waveform'] for item in batch] # List of [1, T]
        targets = torch.stack([item['target'] for item in batch])
        audio_names = [item['audio_name'] for item in batch]
        indices = [item['index'] for item in batch]
        
        # Find max or min length in the batch
        lengths = [w.shape[-1] for w in waveforms]
        
        if mode == "pad":
            target_wave_len = max(lengths)
        elif mode == "truncate":
            target_wave_len = min(lengths)
        else:
            raise ValueError(f"Unknown collate mode: {mode}")
        
        # Pad or Truncate waveforms
        processed_waveforms = []
        for w in waveforms:
            current_len = w.shape[-1]
            if current_len < target_wave_len:
                pad_amount = target_wave_len - current_len
                # Pad at the end
                w_padded = torch.nn.functional.pad(w, (0, pad_amount))
                processed_waveforms.append(w_padded)
            elif current_len > target_wave_len:
                 # Truncate
                 w_truncated = w[..., :target_wave_len]
                 processed_waveforms.append(w_truncated)
            else:
                processed_waveforms.append(w)
                
        processed_waveforms = torch.stack(processed_waveforms)
        
        return {
            "waveform": processed_waveforms,
            "target": targets,
            "audio_name": audio_names,
            "index": indices
        }
