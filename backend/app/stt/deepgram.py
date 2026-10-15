"""Deepgram (cloud) transcription, with a local provider as fallback.

Live calls stream to Deepgram over a WebSocket (see pipeline.DeepgramParticipantStream); file imports
use the pre-recorded REST API. When Deepgram fails, the local fallback provider takes over.
"""
import io
import logging
import wave
from urllib.parse import urlencode

import httpx

from app.config import Settings
from app.stt.base import STTProvider

logger = logging.getLogger("voicebridge")

REST_TIMEOUT_S = 30.0
DIARIZE_TIMEOUT_S = 300.0  # a whole meeting (an hour is ~115 MB of audio)


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return out.getvalue()


class DeepgramSTT:
    def __init__(self, settings: Settings, fallback: STTProvider, transport: httpx.AsyncBaseTransport | None = None):
        self.fallback = fallback
        self.name = f"deepgram (fallback: {fallback.name})"
        self.connect_timeout_s = settings.deepgram_connect_timeout_s
        self.headers = {"Authorization": f"Token {settings.deepgram_api_key}"}
        self._settings = settings
        self._transport = transport
        self._rest_url = settings.deepgram_url.replace("wss://", "https://", 1).replace("ws://", "http://", 1)

    def _common_params(self) -> dict:
        s = self._settings
        return {"model": s.deepgram_model, "language": s.deepgram_language, "smart_format": "true", "punctuate": "true"}

    def live_url(self, diarize: bool = False) -> str:
        """Streaming URL. With `diarize`, every word carries a speaker number (0, 1...) told apart by voice."""
        s = self._settings
        params = self._common_params() | {
            "encoding": "linear16",
            "sample_rate": str(s.sample_rate),
            "channels": "1",
            "interim_results": "true",
            "endpointing": str(s.deepgram_endpointing_ms),
        }
        if diarize:
            params["diarize"] = "true"
        return f"{s.deepgram_url}?{urlencode(params)}"

    async def transcribe(self, pcm: bytes, sample_rate: int, language: str | None = None) -> str:
        """Whole-clip transcription (file import). Falls back to the local provider on any failure."""
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=REST_TIMEOUT_S) as client:
                response = await client.post(
                    self._rest_url,
                    params=self._common_params(),
                    headers=self.headers | {"Content-Type": "audio/wav"},
                    content=pcm_to_wav(pcm, sample_rate),
                )
                response.raise_for_status()
                return response.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
        except Exception as exc:
            logger.warning("Deepgram transcription failed (%s); using %s", exc, self.fallback.name)
            return await self.fallback.transcribe(pcm, sample_rate, language)

    async def diarize(self, pcm: bytes, sample_rate: int) -> list[dict]:
        """Words of a whole recording with who said them: [{start, end, word, speaker}], times in seconds from its
        start. Hearing all of it, Deepgram tells voices apart better than live, where the first minute is unsure.
        Raises on any failure (there is no local fallback for this)."""
        async with httpx.AsyncClient(transport=self._transport, timeout=DIARIZE_TIMEOUT_S) as client:
            response = await client.post(
                self._rest_url,
                params=self._common_params() | {"diarize": "true"},
                headers=self.headers | {"Content-Type": "audio/wav"},
                content=pcm_to_wav(pcm, sample_rate),
            )
            response.raise_for_status()
        words = response.json()["results"]["channels"][0]["alternatives"][0].get("words") or []
        return [
            {"start": w["start"], "end": w["end"], "word": w.get("punctuated_word") or w["word"], "speaker": w.get("speaker", 0)}
            for w in words
        ]
