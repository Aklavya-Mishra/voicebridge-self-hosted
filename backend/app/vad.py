"""Voice activity detection on fixed 512-sample (32 ms at 16 kHz) PCM16 frames."""
from typing import Protocol

import numpy as np

from app.audio_utils import frame_dbfs
from app.config import Settings

FRAME_SAMPLES = 512


class VAD(Protocol):
    frame_samples: int

    def is_speech(self, frame: bytes) -> bool: ...


class EnergyVAD:
    """Loudness threshold, like the dBFS voice check of the reference streaming service."""

    def __init__(self, threshold_dbfs: float = -45.0, frame_samples: int = FRAME_SAMPLES):
        self.threshold_dbfs = threshold_dbfs
        self.frame_samples = frame_samples

    def is_speech(self, frame: bytes) -> bool:
        return frame_dbfs(frame) > self.threshold_dbfs


class SileroVAD:
    """Neural VAD (pip install silero-vad). Stateful, so create one per audio stream."""

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self._model = load_silero_vad()
        self._sample_rate = sample_rate
        self.threshold = threshold
        self.frame_samples = FRAME_SAMPLES

    def is_speech(self, frame: bytes) -> bool:
        audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        prob = self._model(self._torch.from_numpy(audio), self._sample_rate).item()
        return prob >= self.threshold


def make_vad(settings: Settings) -> VAD:
    if settings.vad_provider == "silero":
        return SileroVAD(sample_rate=settings.sample_rate)
    if settings.vad_provider == "energy":
        return EnergyVAD(threshold_dbfs=settings.vad_energy_threshold_dbfs)
    raise ValueError(f"Unknown VAD_PROVIDER: {settings.vad_provider}")
