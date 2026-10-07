"""Key-free provider for development: reports how much speech it received."""
from app.audio_utils import pcm16_duration_s


class MockSTT:
    name = "mock"

    async def transcribe(self, pcm: bytes, sample_rate: int, language: str | None = None) -> str:
        return f"(mock transcript: {pcm16_duration_s(pcm, sample_rate):.1f}s of speech)"
