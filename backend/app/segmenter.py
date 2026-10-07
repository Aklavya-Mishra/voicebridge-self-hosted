"""Splits one speaker's PCM16 mono stream into utterances.

Same idea as the reference streaming service: while there is speech, keep buffering;
once enough trailing silence follows, the sentence is "stable" and gets transcribed as final.
"""
import math
import uuid
from collections import deque
from dataclasses import dataclass

from app.vad import VAD

PARTIAL = "partial"
FINAL = "final"
CANCEL = "cancel"  # an utterance that already showed partials turned out to be noise


@dataclass(frozen=True)
class Segment:
    kind: str
    utterance_id: str
    pcm: bytes
    start_ts: float
    end_ts: float


class Segmenter:
    def __init__(
        self,
        vad: VAD,
        sample_rate: int = 16000,
        end_silence_ms: int = 700,
        max_utterance_ms: int = 15000,
        min_speech_ms: int = 250,
        preroll_ms: int = 200,
        partial_interval_ms: int = 1500,
        partials_enabled: bool = True,
    ):
        self._vad = vad
        self._frame_bytes = vad.frame_samples * 2
        self._frame_s = vad.frame_samples / sample_rate
        frame_ms = self._frame_s * 1000
        self._end_silence_frames = math.ceil(end_silence_ms / frame_ms)
        self._max_frames = math.ceil(max_utterance_ms / frame_ms)
        self._min_speech_frames = math.ceil(min_speech_ms / frame_ms)
        self._partial_frames = math.ceil(partial_interval_ms / frame_ms) if partials_enabled else None
        self._preroll: deque[tuple[bytes, float]] = deque(maxlen=max(1, math.ceil(preroll_ms / frame_ms)))
        self._pending = b""
        self._pending_ts = 0.0
        self._reset_utterance()

    def _reset_utterance(self) -> None:
        self._utterance_id: str | None = None
        self._frames: list[bytes] = []
        self._start_ts = 0.0
        self._end_ts = 0.0
        self._speech_frames = 0
        self._silence_run = 0
        self._frames_since_partial = 0
        self._partial_sent = False

    @property
    def in_utterance(self) -> bool:
        return self._utterance_id is not None

    def feed(self, pcm: bytes, ts: float) -> list[Segment]:
        """Add audio whose first sample was captured at `ts` (seconds, server clock)."""
        if not self._pending:
            self._pending_ts = ts
        self._pending += pcm
        segments: list[Segment] = []
        while len(self._pending) >= self._frame_bytes:
            frame = self._pending[: self._frame_bytes]
            self._pending = self._pending[self._frame_bytes :]
            frame_ts = self._pending_ts
            self._pending_ts += self._frame_s
            segments.extend(self._process(frame, frame_ts))
        return segments

    def flush(self) -> list[Segment]:
        """End of stream: close any open utterance."""
        self._pending = b""
        self._preroll.clear()
        if self.in_utterance:
            return self._finish()
        return []

    def _process(self, frame: bytes, ts: float) -> list[Segment]:
        speech = self._vad.is_speech(frame)
        if not self.in_utterance:
            if not speech:
                self._preroll.append((frame, ts))
                return []
            self._utterance_id = uuid.uuid4().hex[:12]
            self._frames = [f for f, _ in self._preroll]
            self._start_ts = self._preroll[0][1] if self._preroll else ts
            self._preroll.clear()

        self._frames.append(frame)
        self._end_ts = ts + self._frame_s
        if speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
        self._frames_since_partial += 1

        if self._silence_run >= self._end_silence_frames or len(self._frames) >= self._max_frames:
            return self._finish()
        if self._partial_frames and self._frames_since_partial >= self._partial_frames:
            self._frames_since_partial = 0
            self._partial_sent = True
            return [self._segment(PARTIAL)]
        return []

    def _segment(self, kind: str) -> Segment:
        pcm = b"" if kind == CANCEL else b"".join(self._frames)
        return Segment(kind, self._utterance_id, pcm, self._start_ts, self._end_ts)

    def _finish(self) -> list[Segment]:
        if self._speech_frames >= self._min_speech_frames:
            out = [self._segment(FINAL)]
        elif self._partial_sent:
            out = [self._segment(CANCEL)]
        else:
            out = []
        self._reset_utterance()
        return out
