import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import numpy as np
from typing import Optional, List, Dict, Any, Union
import os
from functools import partial
from src.data.audio_utils import resample_and_crop

class AudioSetDataset(Dataset):
    """
    Dataset for AudioSet data stored in HDF5 format.
    
    Args:
        h5_path (str): Path to the HDF5 file containing waveforms and targets.
        exclude_csv_path (Optional[str]): Path to a CSV file containing indices to exclude.
        transform (Optional[callable]): Optional transform to apply to the waveform.
        max_length (Optional[int]): Maximum length of the waveform in samples.
        target_sample_rate (int): Target sample rate for the waveform. Defaults to 32000.
    """
    def __init__(
        self, 
        h5_path: str, 
        exclude_csv_path: Optional[str] = None, 
        transform: Optional[Any] = None, 
        max_length: Optional[int] = None, 
        target_sample_rate: int = 32000
    ):
        self.h5_path = h5_path
        self.transform = transform
        self.max_length = max_length
        self.target_sample_rate = target_sample_rate
        
        # Open HDF5 to get length and metadata
        with h5py.File(h5_path, 'r') as f:
            self.total_length = f['waveform'].shape[0]
            if 'sample_rate' in f.attrs:
                self.source_sample_rate = int(f.attrs['sample_rate'])
            else:
                print(f"Warning: 'sample_rate' attribute not found in {h5_path}. Assuming 32000.")
                self.source_sample_rate = 32000
            
        self.valid_indices = list(range(self.total_length))
        
        if exclude_csv_path and os.path.exists(exclude_csv_path):
            df = pd.read_csv(exclude_csv_path)
            if 'Index' in df.columns:
                exclude_indices = set(df['Index'].values)
                self.valid_indices = [i for i in self.valid_indices if i not in exclude_indices]
            else:
                print(f"Warning: 'Index' column not found in {exclude_csv_path}. No files excluded.")
                
        self.h5_file: Optional[h5py.File] = None

    def _open_h5(self) -> None:
        """Opens the HDF5 file if not already open."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        self._open_h5()
        
        real_idx = self.valid_indices[idx]
        
        # Load waveform
        waveform_int16 = self.h5_file['waveform'][real_idx]
        # Convert to float32 and normalize (16-bit PCM)
        waveform = waveform_int16.astype(np.float32) / 32768.0
        waveform = torch.from_numpy(waveform) # [T]
        
        # Resample and crop
        waveform = resample_and_crop(
            waveform, 
            source_sr=self.source_sample_rate, 
            target_sr=self.target_sample_rate, 
            max_length=self.max_length
        )
        
        # Load target and name
        target = self.h5_file['target'][real_idx]
        audio_name = self.h5_file['audio_name'][real_idx]
        
        target = torch.from_numpy(target).float()
        
        # Add channel dimension: [1, T]
        waveform = waveform.unsqueeze(0)
        
        if self.transform:
            waveform = self.transform(waveform)
            
        return {
            "waveform": waveform,
            "target": target,
            "audio_name": audio_name,
            "index": real_idx
        }

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

class AudioSetDataModule(L.LightningDataModule):
    """
    LightningDataModule for AudioSet.
    
    Args:
        data_dir (str): Root directory for data.
        batch_size (int): Batch size for dataloaders.
        num_workers (int): Number of workers for dataloaders.
        pin_memory (bool): Whether to pin memory in dataloaders.
        train_h5 (str): Filename of training HDF5 file.
        train_csv (str): Filename of training exclusion CSV.
        val_h5 (str): Filename of validation HDF5 file.
        val_csv (str): Filename of validation exclusion CSV.
        max_audio_length_sec (Optional[float]): Maximum audio length in seconds.
        hop_length (Optional[int]): Hop length for spectrogram (samples).
        hop_length_ms (Optional[float]): Hop length in milliseconds.
        patch_size (tuple[int, int]): Patch size (freq, time).
        target_sample_rate (int): Target sample rate.
    """
    def __init__(
        self,
        data_dir: str = "data/AudioSet",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_h5: str = "balanced_train_soxrhq.h5",
        train_csv: str = "silent_files_balanced_train_soxrhq.csv",
        val_h5: str = "eval_soxrhq.h5",
        val_csv: str = "silent_files_eval_soxrhq.csv",
        max_audio_length_sec: Optional[float] = 10.0,
        target_sample_rate: int = 16000,
        collate_mode: str = "pad",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_audio_length_sec = max_audio_length_sec
        self.target_sample_rate = target_sample_rate
        
        if max_audio_length_sec is not None:
            self.max_audio_length = int(max_audio_length_sec * target_sample_rate)
        else:
            self.max_audio_length = None
        self.collate_mode = collate_mode
        
        self.train_h5_path = os.path.join(data_dir, train_h5)
        self.train_csv_path = os.path.join(data_dir, train_csv)
        self.val_h5_path = os.path.join(data_dir, val_h5)
        self.val_csv_path = os.path.join(data_dir, val_csv)
        
        self.train_dataset: Optional[AudioSetDataset] = None
        self.val_dataset: Optional[AudioSetDataset] = None
        self.test_dataset: Optional[AudioSetDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = AudioSetDataset(
                self.train_h5_path,
                self.train_csv_path,
                max_length=self.max_audio_length,
                target_sample_rate=self.target_sample_rate
            )
            self.val_dataset = AudioSetDataset(
                self.val_h5_path,
                self.val_csv_path,
                max_length=self.max_audio_length,
                target_sample_rate=self.target_sample_rate
            )

        if stage == "test":
            self.test_dataset = AudioSetDataset(
                self.val_h5_path,
                self.val_csv_path,
                max_length=self.max_audio_length,
                target_sample_rate=self.target_sample_rate
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

