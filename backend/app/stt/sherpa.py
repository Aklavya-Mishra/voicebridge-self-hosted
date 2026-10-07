"""Streaming transcription with sherpa-onnx (local, free, CPU).

Unlike Whisper, a streaming model emits words while the person is still speaking (~0.4 s behind),
and detects the end of a sentence itself. An optional second pass (Whisper) re-transcribes each
finished sentence for accuracy and punctuation.
"""
import asyncio
import glob
import logging
import os
import re
from dataclasses import dataclass

import numpy as np

from app.config import Settings
from app.stt.base import STTProvider

logger = logging.getLogger("voicebridge")

# Audio appended at the end of a stream so the model emits its last word.
TAIL_PADDING_S = 0.5


@dataclass(frozen=True)
class StreamResult:
    text: str
    timestamps: list[float]  # token start times, seconds since the stream (re)started
    endpoint: bool


def format_text(text: str) -> str:
    """Some models output ALL CAPS without punctuation; make that readable."""
    text = " ".join(text.split())
    if text.isupper():
        text = re.sub(r"\bi\b", "I", text.lower())
    return text[:1].upper() + text[1:]


def _model_file(model_dir: str, part: str) -> str:
    files = sorted(glob.glob(os.path.join(model_dir, f"{part}*.onnx")))
    int8 = [f for f in files if ".int8." in f]
    if not files:
        raise FileNotFoundError(f"No {part}*.onnx in {model_dir}; see README for the model download")
    return (int8 or files)[0]


class SherpaSTT:
    def __init__(self, settings: Settings, refiner: STTProvider | None = None):
        import sherpa_onnx

        model_dir = settings.sherpa_model_dir
        logger.info("Loading sherpa-onnx streaming model from %s", model_dir)
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=os.path.join(model_dir, "tokens.txt"),
            encoder=_model_file(model_dir, "encoder"),
            decoder=_model_file(model_dir, "decoder"),
            joiner=_model_file(model_dir, "joiner"),
            num_threads=settings.sherpa_num_threads,
            sample_rate=settings.sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            # rule1: silence with nothing recognised; rule2: silence after speech ends a sentence.
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=settings.end_silence_ms / 1000,
            rule3_min_utterance_length=settings.max_utterance_ms / 1000,
        )
        self._sample_rate = settings.sample_rate
        self.refiner = refiner
        self.name = f"sherpa+{refiner.name}" if refiner else "sherpa"

    def open_stream(self):
        return self._recognizer.create_stream()

    def decode(self, stream, pcm: bytes) -> StreamResult:
        """Feed PCM16 audio and return the text so far. CPU-bound: call from a worker thread."""
        stream.accept_waveform(self._sample_rate, np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0)
        return self._result(stream, endpoint=None)

    def finish(self, stream) -> StreamResult:
        """End of audio: flush the model and return the last text."""
        stream.accept_waveform(self._sample_rate, np.zeros(int(TAIL_PADDING_S * self._sample_rate), dtype=np.float32))
        stream.input_finished()
        return self._result(stream, endpoint=True)

    def reset(self, stream) -> None:
        self._recognizer.reset(stream)

    def _result(self, stream, endpoint: bool | None) -> StreamResult:
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
        result = self._recognizer.get_result_all(stream)
        is_endpoint = self._recognizer.is_endpoint(stream) if endpoint is None else endpoint
        return StreamResult(format_text(result.text), list(result.timestamps), is_endpoint)

    async def transcribe(self, pcm: bytes, sample_rate: int, language: str | None = None) -> str:
        """Whole-clip transcription (used by file import)."""
        if self.refiner:
            return await self.refiner.transcribe(pcm, sample_rate, language)
        return await asyncio.to_thread(self._transcribe_clip, pcm)

    def _transcribe_clip(self, pcm: bytes) -> str:
        stream = self.open_stream()
        self.decode(stream, pcm)
        return self.finish(stream).text
