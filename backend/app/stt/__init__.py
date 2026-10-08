import logging
from dataclasses import replace

from app.config import Settings
from app.stt.base import STTProvider, clean_transcript

logger = logging.getLogger("voicebridge")


def make_stt(settings: Settings) -> STTProvider:
    if settings.stt_provider == "deepgram":
        if settings.stt_fallback == "deepgram":
            raise ValueError("STT_FALLBACK must be a local provider, not deepgram")
        fallback = make_stt(replace(settings, stt_provider=settings.stt_fallback))
        if not settings.deepgram_api_key:
            logger.warning("DEEPGRAM_API_KEY is not set; using %s only", fallback.name)
            return fallback
        from app.stt.deepgram import DeepgramSTT

        return DeepgramSTT(settings, fallback)
    if settings.stt_provider == "mock":
        from app.stt.mock import MockSTT

        return MockSTT()
    if settings.stt_provider == "whisper":
        from app.stt.whisper import WhisperSTT

        return WhisperSTT(settings)
    raise ValueError(f"Unknown STT_PROVIDER: {settings.stt_provider}")


__all__ = ["STTProvider", "clean_transcript", "make_stt"]
