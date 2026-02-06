from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torchaudio


class DatasetResamplerCropper:
    """
    Resample and optionally crop a waveform.
    Maintains a cache of resamplers for different source sampling rates to optimize instantiation.

    Args:
        target_sr (int): Target sampling rate.
        max_length (Optional[int]): Maximum length in samples (at target_sr).
    """

    def __init__(self, target_sr: int, max_length: Optional[int] = None):
        self.target_sr = target_sr
        self.max_length = max_length
        self.resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def forward(self, waveform: torch.Tensor, source_sr: int) -> torch.Tensor:
        """
        Args:
            waveform (torch.Tensor): Tensor of shape [T] or [C, T].
            source_sr (int): Source sampling rate.

        Returns:
            torch.Tensor: Processed waveform tensor.
        """
        # Resampling and Cropping Logic
        if source_sr != self.target_sr:
            # We need to resample.
            # Optimization: Crop in source domain first if we have a max_length

            if self.max_length is not None:
                # Calculate required source length to get max_length in target domain
                # Add a small buffer to avoid rounding issues
                crop_len_source = round(self.max_length * source_sr / self.target_sr)

                if waveform.shape[-1] > crop_len_source:
                    max_start = waveform.shape[-1] - crop_len_source
                    start = np.random.randint(0, max_start + 1)
                    waveform = waveform[..., start : start + crop_len_source]

            # Resample
            if source_sr not in self.resamplers:
                self.resamplers[source_sr] = torchaudio.transforms.Resample(
                    source_sr, self.target_sr
                )
            resampler = self.resamplers[source_sr]
            waveform = resampler(waveform)

            # Now handle max_length (trim if we cropped with buffer, or if it was already long enough)
            if self.max_length is not None and waveform.shape[-1] > self.max_length:
                waveform = waveform[..., : self.max_length]

        else:
            # No resampling, just standard random crop
            if self.max_length is not None and waveform.shape[-1] > self.max_length:
                max_start = waveform.shape[-1] - self.max_length
                start = np.random.randint(0, max_start + 1)
                waveform = waveform[..., start : start + self.max_length]

        return waveform

    def __call__(self, waveform: torch.Tensor, source_sr: int) -> torch.Tensor:
        return self.forward(waveform, source_sr)


def collate_audio_batch(
    batch: List[Dict[str, Any]],
    waveform_key: str = "waveform",
    mode: str = "pad",  # "pad" or "truncate"
    stack_waveforms: bool = True,
    pad_value: float = 0.0,
    include_keys: Optional[Sequence[str]] = None,
    exclude_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Generic collate function for audio batches where each sample is a dict
    containing at least `waveform_key` with shape [1, T] or [T].

    Pads or truncates waveforms across the batch, and returns a dict that:
        - always includes waveform_key -> Tensor [B, 1, T']
        - includes other keys aggregated into lists (or stacked if possible)

    Parameters
    ----------
    batch:
        List of sample dictionaries.
    waveform_key:
        Key of waveform in sample dict.
    mode:
        "pad" -> pad shorter waveforms to max length in batch
        "truncate" -> truncate longer waveforms to min length in batch
    stack_waveforms:
        If True, returns waveforms stacked into a single tensor [B, 1, T'].
    pad_value:
        Value used for padding.
    include_keys:
        If provided, only these keys will be included in the output (plus waveform_key).
    exclude_keys:
        If provided, these keys will not be included (except waveform_key is always kept).

    Returns
    -------
    Dict[str, Any]
        Collated batch dict.
    """
    if len(batch) == 0:
        raise ValueError("Empty batch passed to collate_audio_batch")

    # 1) Collect waveforms
    waveforms = []
    for item in batch:
        if waveform_key not in item:
            raise KeyError(
                f"Missing key '{waveform_key}' in batch item: {list(item.keys())}"
            )

        w = item[waveform_key]

        if not torch.is_tensor(w):
            raise TypeError(
                f"Expected waveform tensor for key '{waveform_key}', got {type(w)}"
            )

        # Accept [T] or [1, T]
        if w.ndim == 1:
            w = w.unsqueeze(0)
        elif w.ndim != 2:
            raise ValueError(
                f"Expected waveform with shape [T] or [1, T], got {tuple(w.shape)}"
            )

        waveforms.append(w)

    lengths = [w.shape[-1] for w in waveforms]

    # 2) Determine target length
    if mode == "pad":
        target_len = max(lengths)
    elif mode == "truncate":
        target_len = min(lengths)
    else:
        raise ValueError(f"Unknown mode '{mode}' (expected 'pad' or 'truncate')")

    # 3) Pad/truncate each waveform
    processed = []
    for w in waveforms:
        cur_len = w.shape[-1]
        if cur_len < target_len:
            pad_amount = target_len - cur_len
            w2 = torch.nn.functional.pad(w, (0, pad_amount), value=pad_value)
            processed.append(w2)
        elif cur_len > target_len:
            processed.append(w[..., :target_len])
        else:
            processed.append(w)

    if stack_waveforms:
        waveform_batch = torch.stack(processed, dim=0)  # [B, 1, T']
    else:
        waveform_batch = processed  # list of [1, T']

    # 4) Decide which other keys to include
    all_keys = set(batch[0].keys())
    all_keys.add(waveform_key)

    if include_keys is not None:
        keys_to_collate = set(include_keys) | {waveform_key}
    else:
        keys_to_collate = set(all_keys)

    if exclude_keys is not None:
        keys_to_collate -= set(exclude_keys)
        keys_to_collate.add(waveform_key)  # waveform always kept

    # 5) Collate other keys (best effort)
    out: Dict[str, Any] = {waveform_key: waveform_batch}

    for k in keys_to_collate:
        if k == waveform_key:
            continue

        values = [item.get(k, None) for item in batch]

        # If all are tensors of same shape -> stack
        if all(torch.is_tensor(v) for v in values):
            try:
                out[k] = torch.stack(values, dim=0)
                continue
            except Exception:
                # fallback to list if stacking fails
                out[k] = values
                continue

        # If all are numbers (int/float) -> tensor
        if all(isinstance(v, (int, float)) for v in values):
            out[k] = torch.tensor(values)
            continue

        # Otherwise -> list
        out[k] = values

    return out
