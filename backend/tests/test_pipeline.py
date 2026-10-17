import asyncio

import numpy as np

from app.pipeline import ParticipantStream
from app.segmenter import Segmenter
from app.vad import EnergyVAD
from tests.conftest import SR, RecordingSTT, chunks, silence, tone

SPEAKER = {"id": "a", "name": "Asha"}


def build(stt, partials=False):
    emitted, stored = [], []

    async def emit(payload):
        emitted.append(payload)

    async def on_final(utterance):
        stored.append(utterance)

    stream = ParticipantStream(
        code="ABCDEF",
        speaker=SPEAKER,
        segmenter=Segmenter(EnergyVAD(), sample_rate=SR, partials_enabled=partials, partial_interval_ms=500),
        stt=stt,
        sample_rate=SR,
        language=None,
        semaphore=asyncio.Semaphore(2),
        emit=emit,
        on_final=on_final,
    )
    return stream, emitted, stored


async def push_all(stream, pcm):
    ts = 50.0
    for chunk in chunks(pcm):
        await stream.push(chunk, ts)
        ts += len(chunk) / 2 / SR
    await stream.close()


async def test_final_utterances_emitted_and_stored():
    stream, emitted, stored = build(RecordingSTT(text="hello there"))
    await push_all(stream, tone(1.0) + silence(1.0) + tone(1.0) + silence(1.0))
    finals = [p for p in emitted if p["final"]]
    assert len(finals) == 2 and len(stored) == 2
    assert all(p["speaker"] == SPEAKER and p["text"] == "hello there" and p["code"] == "ABCDEF" for p in finals)
    assert [u.id for u in stored] == [p["utterance_id"] for p in finals]


async def test_partials_never_follow_their_final():
    stream, emitted, _ = build(RecordingSTT(), partials=True)
    await push_all(stream, tone(2.5) + silence(1.0))
    final_index = next(i for i, p in enumerate(emitted) if p["final"])
    assert any(not p["final"] for p in emitted[:final_index])
    assert all(p["final"] for p in emitted[final_index:])


BLOCK_S = 0.5  # one "word" per half second in word_audio()


def word_audio(n: int) -> bytes:
    """n half-second blocks, block k a constant level that WordSTT reads back as the word 'w<k>'."""
    return np.concatenate([np.full(int(BLOCK_S * SR), 2000 + 200 * k, dtype=np.int16) for k in range(n)]).tobytes()


class WordSTT(RecordingSTT):
    """Fake word-timed provider for word_audio(): each run of one level is a word, timed from the clip's start."""

    def __init__(self):
        super().__init__()
        self.word_calls: list[int] = []

    def _words(self, pcm: bytes) -> list[tuple[float, float, str]]:
        levels = np.frombuffer(pcm, dtype=np.int16)[:: SR // 100]  # one sample per 10 ms
        runs: list[list] = []
        for i, level in enumerate(levels):
            if level >= 2000 and (not runs or runs[-1][1] != level):
                runs.append([i / 100, int(level)])
        end = len(pcm) / 2 / SR
        return [(start, runs[j + 1][0] if j + 1 < len(runs) else end, f"w{(level - 2000) // 200}")
                for j, (start, level) in enumerate(runs)]

    async def transcribe(self, pcm, sample_rate, language=None):
        self.calls.append(len(pcm))
        return " ".join(w for _, _, w in self._words(pcm))

    async def transcribe_words(self, pcm, sample_rate, language=None):
        self.word_calls.append(len(pcm))
        return self._words(pcm)


def build_windowed(stt, window_s=2.0, keep_s=1.0):
    emitted, stored = [], []

    async def emit(payload):
        emitted.append(payload)

    async def on_final(utterance):
        stored.append(utterance)

    stream = ParticipantStream(
        code="ABCDEF",
        speaker=SPEAKER,
        segmenter=Segmenter(EnergyVAD(), sample_rate=SR, partials_enabled=True, partial_interval_ms=500),
        stt=stt,
        sample_rate=SR,
        language=None,
        semaphore=asyncio.Semaphore(2),
        emit=emit,
        on_final=on_final,
        partial_window_s=window_s,
        partial_keep_s=keep_s,
    )
    return stream, emitted, stored


async def test_live_text_reads_only_the_newest_audio_and_loses_no_words():
    """Live text used to re-transcribe the whole line every 1.5 s (70% of speech-to-text CPU on a test call)."""
    stt = WordSTT()
    stream, emitted, stored = build_windowed(stt)
    ts = 50.0
    for chunk in chunks(silence(0.5) + word_audio(16) + silence(1.5)):
        await stream.push(chunk, ts)
        ts += len(chunk) / 2 / SR
        await asyncio.sleep(0.01)  # let each live update finish, as real time does
    await stream.close()

    partials = [p["text"].split() for p in emitted if not p["final"]]
    assert len(partials) >= 10
    for words in partials:  # every live text is w0 w1 w2... with no word lost or doubled
        assert words == [f"w{i}" for i in range(len(words))]
    assert max(stt.word_calls) <= (2.0 + 0.5 + 0.1) * SR * 2  # never more than window + one interval
    assert len(stored) == 1 and stored[0].text == " ".join(f"w{i}" for i in range(16))  # final: the whole line
    assert max(stt.calls) >= 8 * SR * 2


async def test_without_a_window_live_text_reads_the_whole_line():
    stt = WordSTT()
    stream, emitted, _ = build_windowed(stt, window_s=0)
    await push_all(stream, silence(0.5) + word_audio(8) + silence(1.5))
    assert stt.word_calls == [] and any(not p["final"] for p in emitted)


class SlowSTT(RecordingSTT):
    async def transcribe(self, pcm, sample_rate, language=None):
        await asyncio.sleep(0.3)
        return await super().transcribe(pcm, sample_rate, language)


async def test_final_survives_cancelled_close():
    # A client that stops the mic and disconnects at once cancels the server's stop handler.
    # The last sentence must still be transcribed, stored and emitted.
    stream, emitted, stored = build(SlowSTT(text="last words"))
    for chunk in chunks(tone(1.0)):
        await stream.push(chunk, 0.0)
    closing = asyncio.create_task(stream.close())
    await asyncio.sleep(0.05)
    closing.cancel()
    await asyncio.sleep(0.6)
    assert [u.text for u in stored] == ["last words"]
    assert [p["text"] for p in emitted if p["final"]] == ["last words"]


async def test_stt_failure_gives_empty_final_and_nothing_stored():
    stream, emitted, stored = build(RecordingSTT(fail=True))
    await push_all(stream, tone(1.0) + silence(1.0))
    assert [p["text"] for p in emitted if p["final"]] == [""]
    assert stored == []


async def test_noise_only_transcript_is_dropped():
    stream, emitted, stored = build(RecordingSTT(text="[NOISE]"))
    await push_all(stream, tone(1.0) + silence(1.0))
    assert [p["text"] for p in emitted] == [""]
    assert stored == []
