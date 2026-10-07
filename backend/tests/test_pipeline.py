import asyncio

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
