"""Compare speech-to-text engines on recordings with known words: word error rate (WER) and speed.

Usage (from backend/):
  python tools/eval_stt.py [--engines sherpa,whisper,deepgram] [--diarize]

Test set: tests/data/*.wav with tests/data/trans.txt ("<file> <words>" per line), and each channel of
tools/samples/tts/call_stereo.wav against tools/samples/tts/script.json (make it with tools/make_test_audio.py).
--diarize also scores in-room speaker separation on tools/samples/tts/meeting_mono.wav: the share of words given
to the right person. deepgram needs DEEPGRAM_API_KEY (paid, cloud); the others run locally.
RTF = processing time / audio length (0.1 = ten times faster than real time).
"""
import argparse
import asyncio
import json
import re
import sys
import time
import wave
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402

SR = 16000
NUMBERS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
           "eight": "8", "nine": "9", "ten": "10"}


def normalize(text: str) -> list[str]:
    """Lower case, no punctuation, small numbers as digits, so "Three" and "3" or "fault." and "fault" match."""
    words = re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("-", " ")).split()
    return [NUMBERS.get(w, w) for w in words]


def wer(ref: str, hyp: str) -> tuple[int, int]:
    """(word edits, reference words): word-level Levenshtein distance."""
    r, h = normalize(ref), normalize(hyp)
    row = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, row[0] = row[0], i
        for j, hw in enumerate(h, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (rw != hw))
    return row[len(h)], len(r)


def read_wav(path: Path) -> tuple[list[bytes], float]:
    """Channels of a 16 kHz PCM16 wav, and its length in seconds."""
    import numpy as np

    with wave.open(str(path)) as w:
        if w.getframerate() != SR or w.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16 kHz 16-bit audio")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    return [data[:, c].copy().tobytes() for c in range(data.shape[1])], len(data) / SR


def test_set(root: Path) -> list[tuple[str, bytes, float, str]]:
    """(name, pcm, seconds, reference words)."""
    items = []
    data = root / "tests" / "data"
    trans = data / "trans.txt"
    if trans.exists():
        for line in trans.read_text().splitlines():
            name, _, ref = line.partition(" ")
            (pcm,), seconds = read_wav(data / name)
            items.append((name, pcm, seconds, ref))
    tts = root / "tools" / "samples" / "tts"
    if (tts / "script.json").exists():
        script = json.loads((tts / "script.json").read_text())
        (advisor, customer), seconds = read_wav(tts / "call_stereo.wav")
        for who, pcm in (("advisor", advisor), ("customer", customer)):
            ref = " ".join(line["text"] for line in script if line["speaker"] == who)
            items.append((f"call_stereo.wav:{who}", pcm, seconds, ref))
    return items


def make(engine: str, settings: Settings):
    if engine == "sherpa":
        from app.stt.sherpa import SherpaSTT

        return SherpaSTT(settings)
    if engine == "whisper":
        from app.stt.whisper import WhisperSTT

        return WhisperSTT(settings)
    if engine == "deepgram":
        if not settings.deepgram_api_key:
            raise SystemExit("deepgram needs DEEPGRAM_API_KEY")
        from app.stt.deepgram import DeepgramSTT
        from app.stt.mock import MockSTT

        return DeepgramSTT(settings, MockSTT())
    raise SystemExit(f"unknown engine {engine}")


def speaker_accuracy(words: list[dict], script: list[dict]) -> float:
    """Share of words on the right person, after matching each found voice to the person it overlaps most."""
    pairs = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        line = next((x for x in script if x["start"] - 0.3 <= mid <= x["end"] + 0.3), None)
        if line:
            pairs.append((w["speaker"], line["speaker"]))
    if not pairs:
        return 0.0
    best = {}
    for (voice, person), _ in Counter(pairs).most_common():
        best.setdefault(voice, person)
    return sum(best[v] == p for v, p in pairs) / len(pairs)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", default="sherpa")
    parser.add_argument("--diarize", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = Settings.from_env()
    items = test_set(root)
    print(f"{len(items)} recordings, {sum(i[2] for i in items):.0f} s of audio\n")
    print(f"{'engine':<10} {'recording':<28} {'WER':>6} {'RTF':>6}")
    for engine in args.engines.split(","):
        settings = replace(base, stt_language="en", deepgram_language="en")
        stt = make(engine, settings)
        edits = words = 0
        busy = audio = 0.0
        for name, pcm, seconds, ref in items:
            t = time.perf_counter()
            hyp = await stt.transcribe(pcm, SR, "en")
            took = time.perf_counter() - t
            e, n = wer(ref, hyp)
            edits, words, busy, audio = edits + e, words + n, busy + took, audio + seconds
            print(f"{engine:<10} {name:<28} {e / n:6.1%} {took / seconds:6.3f}")
        print(f"{engine:<10} {'ALL':<28} {edits / words:6.1%} {busy / audio:6.3f}\n")
        if args.diarize and hasattr(stt, "diarize"):
            tts = root / "tools" / "samples" / "tts"
            (pcm,), seconds = read_wav(tts / "meeting_mono.wav")
            t = time.perf_counter()
            found = await stt.diarize(pcm, SR)
            took = time.perf_counter() - t
            script = json.loads((tts / "script.json").read_text())
            voices = len({w["speaker"] for w in found})
            text = " ".join(w["word"] for w in found)
            e, n = wer(" ".join(line["text"] for line in script), text)
            print(f"{engine:<10} in-room meeting: {voices} voices found (2 real), words on the right person "
                  f"{speaker_accuracy(found, script):.1%}, WER {e / n:.1%}, RTF {took / seconds:.3f}\n")


if __name__ == "__main__":
    asyncio.run(main())
