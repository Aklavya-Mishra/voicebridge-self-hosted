import numpy as np
import pytest

from app.config import Settings

SR = 16000


def tone(seconds: float, amplitude: float = 0.3, freq: float = 220.0) -> bytes:
    t = np.arange(int(seconds * SR)) / SR
    return (np.sin(2 * np.pi * freq * t) * amplitude * 32767).astype(np.int16).tobytes()


def silence(seconds: float) -> bytes:
    return np.zeros(int(seconds * SR), dtype=np.int16).tobytes()


def chunks(pcm: bytes, chunk_s: float = 0.1):
    size = int(chunk_s * SR) * 2
    for i in range(0, len(pcm), size):
        yield pcm[i : i + size]


class RecordingSTT:
    """Fake provider: returns a label with the audio length, records every call."""

    name = "fake"

    def __init__(self, text: str | None = None, fail: bool = False):
        self.calls: list[int] = []
        self._text = text
        self._fail = fail

    async def transcribe(self, pcm: bytes, sample_rate: int, language: str | None = None) -> str:
        self.calls.append(len(pcm))
        if self._fail:
            raise RuntimeError("provider down")
        return self._text if self._text is not None else f"speech {len(pcm) / 2 / sample_rate:.1f}s"


@pytest.fixture
def settings() -> Settings:
    return Settings(jwt_secret="test-secret", partial_interval_ms=500)
