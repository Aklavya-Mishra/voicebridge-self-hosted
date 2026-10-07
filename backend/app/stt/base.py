"""Speech-to-text provider interface and transcript clean-up."""
import re
from typing import Protocol

NOISE_TAGS = ("[NOISE]", "[noise]", "[Noise]")


class STTProvider(Protocol):
    name: str

    async def transcribe(self, pcm: bytes, sample_rate: int, language: str | None = None) -> str: ...


def _collapse_repeated_chars(text: str, max_repeats: int) -> str:
    def repl(match: re.Match) -> str:
        run = match.group(0)
        return run if run.isdigit() else run[:max_repeats]

    return re.sub(r"(.)\1{1,}", repl, text)


def _collapse_repeated_words(text: str, max_repeats: int) -> str:
    result: list[str] = []
    prev, count = None, 0
    for word in text.split():
        core = re.sub(r"^\W+|\W+$", "", word).lower()
        if core.isdigit() or re.search(r"\W+$", word):
            result.append(word)
            prev, count = None, 0
            continue
        count = count + 1 if core == prev else 1
        prev = core
        if count <= max_repeats:
            result.append(word)
    return " ".join(result)


def clean_transcript(text: str | None, max_repeats: int = 3) -> str:
    """Drop noise tags and LLM repetition loops (e.g. 'no no no no no no')."""
    if not text:
        return ""
    for tag in NOISE_TAGS:
        text = text.replace(tag, " ")
    text = " ".join(text.split())
    return _collapse_repeated_words(_collapse_repeated_chars(text, max_repeats), max_repeats).strip()
