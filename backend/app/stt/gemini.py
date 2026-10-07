"""Gemini audio transcription, via a Gemini API key or Vertex AI."""
from google import genai
from google.genai import types

from app.audio_utils import pcm16_to_wav
from app.config import Settings

SYSTEM_INSTRUCTION = (
    "You are an expert audio transcriber. You only transcribe the audio and formulate valid sentences. "
    "Return the entire text without providing any further context or answers. "
    "For audio with noises only return [NOISE]. "
    "STRICTLY do not repeat phrases, words or characters in output if not present in audio."
)


class GeminiSTT:
    name = "gemini"

    def __init__(self, settings: Settings):
        if settings.gemini_api_key:
            self._client = genai.Client(api_key=settings.gemini_api_key)
        elif settings.vertex_project:
            self._client = genai.Client(
                vertexai=True, project=settings.vertex_project, location=settings.vertex_location
            )
        else:
            raise ValueError("STT_PROVIDER=gemini needs GEMINI_API_KEY or VERTEX_PROJECT")
        self._model = settings.gemini_model
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION, temperature=0, top_p=0.95, max_output_tokens=400
        )

    async def transcribe(self, pcm: bytes, sample_rate: int, language: str | None = None) -> str:
        prompt = "Transcribe this audio."
        if language:
            prompt += f" The speech is in {language}."
        audio = types.Part.from_bytes(data=pcm16_to_wav(pcm, sample_rate), mime_type="audio/wav")
        response = await self._client.aio.models.generate_content(
            model=self._model, contents=[prompt, audio], config=self._config
        )
        return response.text or ""
