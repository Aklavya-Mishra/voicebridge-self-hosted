"""PCM16 helpers shared by the live pipeline and the importer."""
import io
import math
import wave

import numpy as np

SILENCE_DBFS = -120.0


def pcm16_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def pcm16_duration_s(pcm: bytes, sample_rate: int) -> float:
    return len(pcm) / 2 / sample_rate


def frame_dbfs(frame: bytes) -> float:
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return SILENCE_DBFS
    rms = float(np.sqrt(np.mean(samples * samples)))
    if rms == 0:
        return SILENCE_DBFS
    return 20 * math.log10(rms / 32768.0)
