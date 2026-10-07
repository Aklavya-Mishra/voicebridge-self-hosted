"""Local transcription with faster-whisper. No API key; the model downloads once on first start."""
import asyncio
import logging
import os

import numpy as np

from app.config import Settings

logger = logging.getLogger("voicebridge")


class WhisperSTT:
    name = "whisper"

    def __init__(self, settings: Settings):
        # OpenMP worker threads busy-wait by default. On some machines (seen on Windows) that makes a
        # 4-thread run ~20x slower than 1 thread. Must be set before ctranslate2 is first imported.
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        from faster_whisper import WhisperModel

        logger.info(
            "Loading faster-whisper model %s on %s (%s), %d worker(s) x %d thread(s)",
            settings.whisper_model,
            settings.whisper_device,
            settings.whisper_compute_type,
            settings.whisper_num_workers,
            settings.whisper_cpu_threads,
        )
        # num_workers lets several transcribe() calls run in parallel from different threads
        # (e.g. both speakers, or a partial and a final), so no lock is needed.
        self._model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            cpu_threads=settings.whisper_cpu_threads,
            num_workers=settings.whisper_num_workers,
        )
        self._beam_size = settings.whisper_beam_size
        # The first inference is several times slower than later ones; pay that cost at startup.
        self._run(np.zeros(16000, dtype=np.int16).tobytes(), "en")

    async def transcribe(self, pcm: bytes, sample_rate: int, language: str | None = None) -> str:
        return await asyncio.to_thread(self._run, pcm, language)

    def _run(self, pcm: bytes, language: str | None) -> str:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            audio,
            language=language,  # ISO code such as "en"; None = auto-detect per utterance
            beam_size=self._beam_size,
            # Each utterance stands alone; feeding previous text causes repetition loops.
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments)
