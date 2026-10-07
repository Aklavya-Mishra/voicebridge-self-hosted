"""Turns two device streams (mic, speaker loopback) into 16 kHz stereo PCM16 chunks for VoiceBridge.

Left = mic (the user), right = loopback (the other person on the call). Loopback can come from several
output devices (speakers, HDMI monitor, headset: wherever the call app plays); they are summed.
The mic drives timing: each chunk takes the same span of each loopback, padded with silence when Windows
delivered none (nothing was playing there) and trimmed when it runs ahead, so the channels stay aligned.
Callbacks from the audio threads call add_mic/add_loop; the sender thread calls chunks().
"""
import threading

import numpy as np
import soxr

OUT_RATE = 16000
CHUNK_FRAMES = 1600  # 100 ms


class _Channel:
    def __init__(self, rate: int, channels: int):
        self._channels = channels
        self._resampler = soxr.ResampleStream(rate, OUT_RATE, 1, dtype="float32")
        self.buffer = np.zeros(0, dtype=np.float32)

    def add(self, samples: np.ndarray) -> None:
        mono = samples.reshape(-1, self._channels).mean(axis=1, dtype=np.float32)
        self.buffer = np.concatenate([self.buffer, self._resampler.resample_chunk(mono)])

    def take(self, n: int) -> np.ndarray:
        out = self.buffer[:n]
        self.buffer = self.buffer[n:]
        return out if len(out) == n else np.concatenate([out, np.zeros(n - len(out), dtype=np.float32)])


def _to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return np.where(x < 0, x * 32768, x * 32767).astype(np.int16)


class StereoMixer:
    def __init__(
        self,
        *,
        mic_rate: int,
        mic_channels: int,
        loop_rate: int,
        loop_channels: int,
        extra_loops: list[tuple[int, int]] = (),  # (rate, channels) of further output devices
        max_lead_s: float = 0.5,
    ):
        self._mic = _Channel(mic_rate, mic_channels)
        self._loops = [_Channel(loop_rate, loop_channels)] + [_Channel(r, c) for r, c in extra_loops]
        self._max_lead = int(max_lead_s * OUT_RATE)
        self._lock = threading.Lock()

    def add_mic(self, samples: np.ndarray) -> None:
        with self._lock:
            self._mic.add(samples)

    def add_loop(self, samples: np.ndarray, source: int = 0) -> None:
        with self._lock:
            self._loops[source].add(samples)

    def loop_backlog_s(self) -> float:
        with self._lock:
            return max(len(loop.buffer) for loop in self._loops) / OUT_RATE

    def chunks(self) -> list[bytes]:
        """Interleaved stereo PCM16 chunks of 100 ms for every full 100 ms of mic audio."""
        out = []
        with self._lock:
            while len(self._mic.buffer) >= CHUNK_FRAMES:
                left = self._mic.take(CHUNK_FRAMES)
                right = sum(loop.take(CHUNK_FRAMES) for loop in self._loops)
                out.append(np.stack([_to_int16(left), _to_int16(right)], axis=1).tobytes())
            for loop in self._loops:
                excess = len(loop.buffer) - self._max_lead
                if excess > 0:
                    loop.buffer = loop.buffer[excess:]  # keep the newest audio
        return out
