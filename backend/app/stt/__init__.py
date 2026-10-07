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
    if settings.stt_provider == "gemini":
        from app.stt.gemini import GeminiSTT

        return GeminiSTT(settings)
    if settings.stt_provider == "whisper":
        from app.stt.whisper import WhisperSTT

        return WhisperSTT(settings)
    if settings.stt_provider == "sherpa":
        from app.stt.sherpa import SherpaSTT

        refiner = None
        if settings.final_pass == "whisper":
            from app.stt.whisper import WhisperSTT

            refiner = WhisperSTT(settings)
        elif settings.final_pass != "none":
            raise ValueError(f"Unknown FINAL_PASS: {settings.final_pass}")
        return SherpaSTT(settings, refiner)
    raise ValueError(f"Unknown STT_PROVIDER: {settings.stt_provider}")


__all__ = ["STTProvider", "clean_transcript", "make_stt"]
