import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

# Ensure local `src` package is importable when run as a script.
sys.path.append(os.path.abspath("."))

import src.data.yt1b_datamodule as yt1b_module
from src.data.yt1b_datamodule import YT1BDataset


class _FakeDecoded:
    def __init__(self, data: torch.Tensor):
        self.data = data


class _FakeMetadata:
    def __init__(
        self,
        duration_seconds: Optional[float],
        duration_seconds_from_header: Optional[float],
    ):
        self.duration_seconds = duration_seconds
        self.duration_seconds_from_header = duration_seconds_from_header


class _TrackingDecoder:
    init_calls: list[dict] = []
    range_calls: list[tuple[float, Optional[float]]] = []
    all_calls: int = 0
    duration_seconds: Optional[float] = 10.0
    duration_seconds_from_header: Optional[float] = None
    output_samples: int = 4000
    return_1d: bool = False

    @classmethod
    def reset(cls) -> None:
        cls.init_calls = []
        cls.range_calls = []
        cls.all_calls = 0

    def __init__(self, source: str, sample_rate: int, num_channels: int):
        self.source = source
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.metadata = _FakeMetadata(
            duration_seconds=self.duration_seconds,
            duration_seconds_from_header=self.duration_seconds_from_header,
        )
        self.__class__.init_calls.append(
            {
                "source": source,
                "sample_rate": sample_rate,
                "num_channels": num_channels,
            }
        )

    def get_all_samples(self) -> _FakeDecoded:
        self.__class__.all_calls += 1
        data = torch.linspace(
            -1.0, 1.0, steps=self.__class__.output_samples, dtype=torch.float32
        )
        if not self.__class__.return_1d:
            data = data.unsqueeze(0)
        return _FakeDecoded(data=data)

    def get_samples_played_in_range(
        self, start_seconds: float = 0.0, stop_seconds: Optional[float] = None
    ) -> _FakeDecoded:
        self.__class__.range_calls.append((float(start_seconds), stop_seconds))

        if stop_seconds is None:
            num_samples = self.__class__.output_samples
        else:
            duration_sec = max(0.0, float(stop_seconds) - float(start_seconds))
            num_samples = max(1, int(round(duration_sec * float(self.sample_rate))))

        data = torch.ones(num_samples, dtype=torch.float32)
        if not self.__class__.return_1d:
            data = data.unsqueeze(0)
        return _FakeDecoded(data=data)


def _write_metadata_parquet(path: Path, duration_sec: float) -> None:
    df = pd.DataFrame(
        {
            "file_path": ["fake_audio.wav"],
            "video_id": ["audio_0"],
            "duration_sec": [duration_sec],
        }
    )
    df.to_parquet(path, index=False)


def _set_audio_decoder(decoder: object, import_error: Optional[Exception]) -> tuple:
    previous_decoder = yt1b_module.AudioDecoder
    previous_error = yt1b_module._TORCHCODEC_IMPORT_ERROR
    yt1b_module.AudioDecoder = decoder
    yt1b_module._TORCHCODEC_IMPORT_ERROR = import_error
    return previous_decoder, previous_error


def test_fail_fast_when_torchcodec_unavailable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        parquet_path = Path(tmp) / "metadata.parquet"
        _write_metadata_parquet(parquet_path, duration_sec=12.0)

        import_error = RuntimeError("missing libtorchcodec")
        old_decoder, old_error = _set_audio_decoder(None, import_error)
        try:
            try:
                YT1BDataset(parquet_path=str(parquet_path))
                raise AssertionError("Expected RuntimeError when AudioDecoder is None")
            except RuntimeError as exc:
                assert "requires torchcodec AudioDecoder" in str(exc)
                assert "installing-torchcodec" in str(exc)
                assert exc.__cause__ is import_error
        finally:
            yt1b_module.AudioDecoder = old_decoder
            yt1b_module._TORCHCODEC_IMPORT_ERROR = old_error


def test_decoder_constructor_and_range_decode() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        parquet_path = Path(tmp) / "metadata.parquet"
        _write_metadata_parquet(parquet_path, duration_sec=12.0)

        _TrackingDecoder.reset()
        _TrackingDecoder.duration_seconds = 10.0
        _TrackingDecoder.duration_seconds_from_header = None
        _TrackingDecoder.return_1d = False

        old_decoder, old_error = _set_audio_decoder(_TrackingDecoder, None)
        try:
            dataset = YT1BDataset(
                parquet_path=str(parquet_path),
                target_sample_rate=22050,
                decode_window_sec=1.5,
            )
            sample = dataset[0]
        finally:
            yt1b_module.AudioDecoder = old_decoder
            yt1b_module._TORCHCODEC_IMPORT_ERROR = old_error

        assert len(_TrackingDecoder.init_calls) == 1
        init_call = _TrackingDecoder.init_calls[0]
        assert init_call["sample_rate"] == 22050
        assert init_call["num_channels"] == 1
        assert _TrackingDecoder.all_calls == 0
        assert len(_TrackingDecoder.range_calls) == 1

        start_sec, stop_sec = _TrackingDecoder.range_calls[0]
        assert stop_sec is not None
        assert 0.0 <= start_sec <= 8.5
        assert abs((stop_sec - start_sec) - 1.5) < 1e-6

        waveform = sample["waveform"]
        assert waveform.ndim == 2
        assert waveform.shape[0] == 1
        assert waveform.dtype == torch.float32


def test_shape_type_and_get_all_samples_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        parquet_path = Path(tmp) / "metadata.parquet"
        _write_metadata_parquet(parquet_path, duration_sec=7.0)

        _TrackingDecoder.reset()
        _TrackingDecoder.return_1d = True
        _TrackingDecoder.output_samples = 1234

        old_decoder, old_error = _set_audio_decoder(_TrackingDecoder, None)
        try:
            dataset = YT1BDataset(
                parquet_path=str(parquet_path),
                decode_window_sec=None,
                max_length=None,
                target_sample_rate=16000,
            )
            sample = dataset[0]
        finally:
            yt1b_module.AudioDecoder = old_decoder
            yt1b_module._TORCHCODEC_IMPORT_ERROR = old_error

        assert _TrackingDecoder.all_calls == 1
        assert len(_TrackingDecoder.range_calls) == 0

        waveform = sample["waveform"]
        assert waveform.ndim == 2
        assert waveform.shape == (1, 1234)
        assert waveform.dtype == torch.float32
        assert sample["audio_name"] == "audio_0"
        assert sample["index"] == 0


def test_auto_decode_window_uses_max_length_and_fallback_duration() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        parquet_path = Path(tmp) / "metadata.parquet"
        _write_metadata_parquet(parquet_path, duration_sec=12.0)

        _TrackingDecoder.reset()
        _TrackingDecoder.duration_seconds = None
        _TrackingDecoder.duration_seconds_from_header = None
        _TrackingDecoder.return_1d = False

        old_decoder, old_error = _set_audio_decoder(_TrackingDecoder, None)
        try:
            dataset = YT1BDataset(
                parquet_path=str(parquet_path),
                max_length=3200,
                target_sample_rate=16000,
                decode_window_sec=None,
            )
            sample = dataset[0]
        finally:
            yt1b_module.AudioDecoder = old_decoder
            yt1b_module._TORCHCODEC_IMPORT_ERROR = old_error

        assert _TrackingDecoder.all_calls == 0
        assert len(_TrackingDecoder.range_calls) == 1

        start_sec, stop_sec = _TrackingDecoder.range_calls[0]
        assert stop_sec is not None
        assert abs((stop_sec - start_sec) - 0.2) < 1e-6

        waveform = sample["waveform"]
        assert waveform.shape == (1, 3200)
        assert waveform.dtype == torch.float32


def verify_yt1b_torchcodec() -> None:
    test_fail_fast_when_torchcodec_unavailable()
    test_decoder_constructor_and_range_decode()
    test_shape_type_and_get_all_samples_path()
    test_auto_decode_window_uses_max_length_and_fallback_duration()
    print("YT1B TorchCodec verification successful.")


if __name__ == "__main__":
    verify_yt1b_torchcodec()
