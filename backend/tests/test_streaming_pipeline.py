import asyncio

from app.pipeline import StreamingParticipantStream
from app.stt.sherpa import StreamResult, format_text
from tests.conftest import SR, RecordingSTT, chunks, silence, tone

SPEAKER = {"id": "b", "name": "Ben"}


class FakeStreamingSTT:
    """Scripted streaming model: one word per loud 100 ms chunk, sentence ends after 3 quiet chunks."""

    name = "fake-streaming"

    def __init__(self, refiner=None):
        self.refiner = refiner
        self.resets = 0

    def open_stream(self):
        return {"words": [], "quiet": 0, "elapsed": 0.0}

    def decode(self, stream, pcm):
        loud = max(pcm[1::2]) > 0
        if loud:
            stream["words"].append((f"w{len(stream['words'])}", stream["elapsed"]))
        stream["quiet"] = 0 if loud else stream["quiet"] + 1
        stream["elapsed"] += len(pcm) / 2 / SR
        return self._result(stream, endpoint=bool(stream["words"]) and stream["quiet"] >= 3)

    def finish(self, stream):
        return self._result(stream, endpoint=True)

    def reset(self, stream):
        self.resets += 1
        stream.update(words=[], quiet=0, elapsed=0.0)

    def _result(self, stream, endpoint):
        return StreamResult(" ".join(w for w, _ in stream["words"]), [t for _, t in stream["words"]], endpoint)


def build(stt):
    emitted, stored = [], []

    async def emit(payload):
        emitted.append(payload)

    async def on_final(utterance):
        stored.append(utterance)

    stream = StreamingParticipantStream(
        code="ABCDEF",
        speaker=SPEAKER,
        stt=stt,
        sample_rate=SR,
        language=None,
        semaphore=asyncio.Semaphore(2),
        emit=emit,
        on_final=on_final,
    )
    return stream, emitted, stored


async def push_all(stream, pcm, close=True):
    ts = 100.0
    for chunk in chunks(pcm):
        await stream.push(chunk, ts)
        ts += len(chunk) / 2 / SR
    if close:
        await stream.close()


async def test_words_stream_as_partials_then_sentence_is_final():
    stream, emitted, stored = build(FakeStreamingSTT())
    await push_all(stream, silence(0.5) + tone(0.3) + silence(0.5))
    texts = [(p["text"], p["final"]) for p in emitted]
    assert texts == [("w0", False), ("w0 w1", False), ("w0 w1 w2", False), ("w0 w1 w2", True)]
    assert len({p["utterance_id"] for p in emitted}) == 1
    assert [u.text for u in stored] == ["w0 w1 w2"]
    # start is the first word's time, not the start of the leading silence
    assert abs(stored[0].start_ts - 100.5) < 1e-6


async def test_each_sentence_gets_its_own_utterance():
    stt = FakeStreamingSTT()
    stream, emitted, stored = build(stt)
    await push_all(stream, tone(0.2) + silence(0.4) + tone(0.2) + silence(0.4))
    assert [u.text for u in stored] == ["w0 w1", "w0 w1"]
    assert stored[0].id != stored[1].id and stored[1].start_ts > stored[0].start_ts
    assert stt.resets == 2


async def test_close_finalises_an_open_sentence():
    stream, emitted, stored = build(FakeStreamingSTT())
    await push_all(stream, tone(0.2))
    assert [u.text for u in stored] == ["w0 w1"]
    assert emitted[-1]["final"]


def seconds(nbytes: int) -> float:
    return nbytes / 2 / SR


async def test_refiner_replaces_streaming_text_with_a_clip_around_the_words():
    refiner = RecordingSTT(text="Hello there.")
    stream, emitted, stored = build(FakeStreamingSTT(refiner))
    await push_all(stream, silence(1.0) + tone(0.2) + silence(0.4))
    finals = [p for p in emitted if p["final"]]
    assert [p["text"] for p in finals] == ["Hello there."] and [u.text for u in stored] == ["Hello there."]
    assert all(not p["final"] for p in emitted[:-1])
    # words at 1.0 and 1.1 s: clip runs from 0.5 s before the first to 0.35 s after the last
    assert [round(seconds(n), 2) for n in refiner.calls] == [0.95]


async def test_refiner_clips_do_not_overlap_and_cross_the_reset():
    refiner = RecordingSTT(text="x")
    stream, emitted, stored = build(FakeStreamingSTT(refiner))
    await push_all(stream, tone(0.2) + silence(0.3) + tone(0.2) + silence(0.3))
    # sentence 1: 0.0-0.45 s; sentence 2 starts where sentence 1 ended (its lead reaches back
    # before the model reset at 0.5 s) and ends 0.35 s after its last word at 0.6 s
    assert [round(seconds(n), 2) for n in refiner.calls] == [0.45, 0.5]
    assert stored[0].end_ts <= stored[1].start_ts


async def test_refiner_failure_keeps_streaming_text():
    stream, emitted, stored = build(FakeStreamingSTT(RecordingSTT(fail=True)))
    await push_all(stream, tone(0.2) + silence(0.4))
    assert [u.text for u in stored] == ["w0 w1"]


async def test_silence_only_emits_nothing():
    stream, emitted, stored = build(FakeStreamingSTT())
    await push_all(stream, silence(2.0))
    assert emitted == [] and stored == []


def test_format_text_fixes_all_caps():
    assert format_text("HELLO I WANT TO   GO") == "Hello I want to go"
    assert format_text("already Fine, thanks.") == "Already Fine, thanks."
    assert format_text("") == ""
