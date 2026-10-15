"""Deepgram provider against a local fake Deepgram server: no network, no credit used."""
import asyncio
import json
from http import HTTPStatus

import httpx
import pytest
from websockets.asyncio.server import serve

from app.config import Settings
from app.pipeline import DeepgramParticipantStream
from app.stt.deepgram import DeepgramSTT
from tests.conftest import SR, RecordingSTT, chunks, silence, tone

SPEAKER = {"id": "a", "name": "Aklavya"}
T0 = 1000.0  # server clock of the first audio sample


def results(
    text: str,
    start: float,
    duration: float,
    *,
    is_final: bool,
    speech_final: bool = False,
    words_from: float | None = None,
    speakers: list[int] | None = None,
) -> str:
    """`words_from`: the words are spread from this time to the end of the result (default: its start).
    `speakers`: each word's voice number (diarization); none by default."""
    words = text.split()
    words_start = start if words_from is None else words_from
    step = (start + duration - words_start) / max(1, len(words))
    return json.dumps(
        {
            "type": "Results",
            "start": start,
            "duration": duration,
            "is_final": is_final,
            "speech_final": speech_final,
            "channel": {
                "alternatives": [
                    {
                        "transcript": text,
                        "words": [
                            {"word": w, "start": words_start + i * step + 0.1, "end": words_start + (i + 1) * step - 0.1}
                            | ({"speaker": speakers[i]} if speakers else {})
                            for i, w in enumerate(words)
                        ],
                    }
                ]
            },
        }
    )


class FakeDeepgram:
    """Scripted server. `script(ws, seconds_received)` runs after each audio message.

    `accept` limits how many connections are accepted; later ones get 401 (as with a bad key).
    """

    def __init__(self, script, accept: int = 99):
        self.script = script
        self.accept = accept
        self.connections = 0
        self.received: list[bytes] = []
        self.params: dict = {}
        self.auth = ""

    def _process_request(self, connection, request):
        self.connections += 1
        self.auth = request.headers.get("Authorization", "")
        self.params = dict(p.split("=", 1) for p in request.path.split("?", 1)[1].split("&"))
        if self.connections > self.accept:
            return connection.respond(HTTPStatus.UNAUTHORIZED, "bad key\n")
        return None

    async def _handler(self, ws):
        received = 0.0
        async for message in ws:
            if isinstance(message, str):
                if json.loads(message).get("type") == "CloseStream":
                    await self.script(ws, None)
                    break
                continue
            self.received.append(message)
            received += len(message) / 2 / SR
            if await self.script(ws, round(received, 3)) == "drop":
                await ws.close(1011, "server error")
                return
        await ws.close()

    async def __aenter__(self):
        self._server = await serve(self._handler, "127.0.0.1", 0, process_request=self._process_request)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/v1/listen"
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        await self._server.wait_closed()


class FakeLocalStream:
    def __init__(self):
        self.pushed: list[tuple[bytes, float]] = []
        self.closed = False

    async def push(self, pcm: bytes, ts: float) -> None:
        self.pushed.append((pcm, ts))

    async def close(self) -> None:
        self.closed = True


def build(url: str, api_key: str = "test-key", speaker_for=None):
    settings = Settings(deepgram_api_key=api_key, deepgram_url=url, deepgram_connect_timeout_s=2.0)
    stt = DeepgramSTT(settings, fallback=RecordingSTT("local text"))
    emitted, stored, notices, fallbacks, starts = [], [], [], [], []

    async def emit(payload):
        emitted.append(payload)

    async def on_final(utterance):
        stored.append(utterance)

    async def notify(payload):
        notices.append(payload)

    async def on_start(ts):
        starts.append(ts)

    def make_fallback():
        fallbacks.append(FakeLocalStream())
        return fallbacks[-1]

    stream = DeepgramParticipantStream(
        code="ABCDEF",
        speaker=SPEAKER,
        stt=stt,
        sample_rate=SR,
        emit=emit,
        on_final=on_final,
        notify=notify,
        make_fallback=make_fallback,
        on_start=on_start,
        speaker_for=speaker_for,
    )
    build.starts = starts
    return stream, emitted, stored, notices, fallbacks


async def feed(stream, pcm: bytes, start_ts: float = T0) -> None:
    for i, chunk in enumerate(chunks(pcm)):
        await stream.push(chunk, start_ts + i * 0.1)
        await asyncio.sleep(0.005)  # let server replies arrive between chunks


async def test_live_partials_then_final_sentence():
    async def script(ws, received):
        if received == 1.0:
            await ws.send(results("hello", 0.0, 1.0, is_final=False))
        elif received == 2.0:
            await ws.send(results("hello world", 0.0, 2.0, is_final=True, speech_final=True))

    async with FakeDeepgram(script) as server:
        stream, emitted, stored, notices, fallbacks = build(server.url)
        await feed(stream, tone(2.0))
        await stream.close()

    assert server.auth == "Token test-key"
    assert server.params["model"] == "nova-3" and server.params["language"] == "multi"
    assert server.params["encoding"] == "linear16" and server.params["sample_rate"] == str(SR)
    partials = [e for e in emitted if not e["final"]]
    assert partials and partials[0]["text"] == "hello"
    assert [u.text for u in stored] == ["hello world"]
    assert stored[0].speaker_name == "Aklavya"
    assert stored[0].start_ts == pytest.approx(T0 + 0.1, abs=0.01)
    assert stored[0].end_ts == pytest.approx(T0 + 1.9, abs=0.01)
    finals = [e for e in emitted if e["final"]]
    assert [f["text"] for f in finals] == ["hello world"]
    assert finals[0]["utterance_id"] == partials[0]["utterance_id"]
    assert not notices and not fallbacks


async def test_short_pauses_stay_in_one_bubble():
    """A pause Deepgram ends a sentence on (speech_final) is not a new bubble: only 3 s of silence is."""
    async def script(ws, received):
        if received == 1.0:
            await ws.send(results("hello", 0.0, 1.0, is_final=True, speech_final=True))
        elif received == 2.5:
            await ws.send(results("world", 1.5, 1.0, is_final=True, speech_final=True))

    async with FakeDeepgram(script) as server:
        stream, emitted, stored, *_ = build(server.url)
        await feed(stream, tone(2.5))
        await stream.close()

    assert [u.text for u in stored] == ["hello world"]


async def test_late_deepgram_results_do_not_split_a_short_pause():
    """Seen live: Deepgram answers 2-3 s behind the audio sent. A 1.5 s pause must not look like 3 s just
    because the next words have not come back yet."""
    async def script(ws, received):
        if received == 1.0:
            await ws.send(results("hello", 0.0, 1.0, is_final=True, speech_final=True))
        elif received == 4.5:  # "world" was spoken at 2.5 s; Deepgram reports it late
            await ws.send(results("world", 2.5, 1.0, is_final=True, speech_final=True))

    async with FakeDeepgram(script) as server:
        stream, emitted, stored, *_ = build(server.url)
        await feed(stream, tone(1.0) + silence(1.5) + tone(1.0) + silence(1.5))
        await stream.close()

    assert [u.text for u in stored] == ["hello world"]


async def test_three_seconds_of_silence_ends_the_bubble():
    async def script(ws, received):
        if received == 1.0:
            await ws.send(results("hello", 0.0, 1.0, is_final=True, speech_final=True))
        elif received == 4.5:  # Deepgram reports the silence it processed (an empty result)
            await ws.send(results("", 1.0, 3.5, is_final=True))
        elif received == 5.5:
            await ws.send(results("again", 5.0, 0.5, is_final=True, speech_final=True))

    async with FakeDeepgram(script) as server:
        stream, emitted, stored, *_ = build(server.url)
        await feed(stream, tone(1.0) + silence(3.5))
        await asyncio.sleep(0.1)  # the reply to the last chunk
        assert [u.text for u in stored] == ["hello"]  # stored once 3 s passed, before the next words
        await feed(stream, tone(1.0), start_ts=T0 + 4.5)
        await stream.close()

    assert [u.text for u in stored] == ["hello", "again"]
    assert stored[0].id != stored[1].id


async def test_finalized_text_without_pause_is_stored_on_close():
    async def script(ws, received):
        if received == 1.0:
            await ws.send(results("good morning", 0.0, 1.0, is_final=True))

    async with FakeDeepgram(script) as server:
        stream, emitted, stored, *_ = build(server.url)
        await feed(stream, tone(1.5))
        await stream.close()

    assert [u.text for u in stored] == ["good morning"]


async def test_words_shown_live_but_missing_from_the_final_are_kept():
    # Seen on real calls: Deepgram's final for a sentence ending in a number held only the number.
    async def script(ws, received):
        if received == 1.0:
            await ws.send(results("So, my credit card number is", 0.0, 1.0, is_final=False))
        elif received == 2.0:
            await ws.send(results("So, my credit card number is eight seven", 0.0, 2.0, is_final=False))
        elif received == 3.0:
            await ws.send(results("87548734.", 0.0, 3.0, is_final=True, speech_final=True, words_from=1.4))

    async with FakeDeepgram(script) as server:
        stream, emitted, stored, *_ = build(server.url)
        await feed(stream, tone(3.0))
        await stream.close()

    assert [u.text for u in stored] == ["So, my credit card number is 87548734."]
    assert stored[0].start_ts == pytest.approx(T0 + 0.1, abs=0.01)


async def test_other_speaker_cuts_the_open_sentence_and_the_rest_continues_in_a_new_bubble():
    async def script(ws, received):
        if received == 2.0:
            await ws.send(results("I was calling about my booking for next week", 0.0, 2.0, is_final=False))
        elif received == 2.5:
            await ws.send(results("I was calling about my booking for next week", 0.0, 2.0, is_final=True))
        elif received == 3.0:
            await ws.send(results("please", 2.0, 1.0, is_final=True, speech_final=True))

    async with FakeDeepgram(script) as server:
        stream, emitted, stored, *_ = build(server.url)
        await feed(stream, tone(2.0))
        await asyncio.sleep(0.05)
        await stream.cut(T0 + 1.2)  # the other speaker started talking 1.2 s in
        await feed(stream, tone(1.0), start_ts=T0 + 2.0)
        await stream.close()

    assert [u.text for u in stored] == ["I was calling about my", "booking for next week please"]
    assert stored[0].start_ts == pytest.approx(T0 + 0.1, abs=0.01)
    assert stored[1].start_ts >= T0 + 1.2
    finals = [e for e in emitted if e["final"]]
    assert [f["text"] for f in finals] == ["I was calling about my", "booking for next week please"]
    assert finals[0]["utterance_id"] != finals[1]["utterance_id"]
    # Only the speaker's own new sentence is announced; the continuation after the cut is not.
    assert build.starts == [pytest.approx(T0 + 0.1, abs=0.01)]


async def test_cut_before_the_sentence_started_changes_nothing():
    async def script(ws, received):
        if received == 1.0:
            await ws.send(results("hello there", 0.5, 0.5, is_final=True, speech_final=True))

    async with FakeDeepgram(script) as server:
        stream, emitted, stored, *_ = build(server.url)
        await stream.cut(T0 + 0.2)
        await feed(stream, tone(1.0))
        await stream.close()

    assert [u.text for u in stored] == ["hello there"]


async def test_rejected_key_falls_back_to_local_with_all_audio():
    async def script(ws, received):
        pass

    async with FakeDeepgram(script, accept=0) as server:
        stream, emitted, stored, notices, fallbacks = build(server.url, api_key="wrong")
        audio = tone(1.0)
        await feed(stream, audio)
        await stream.close()

    assert len(fallbacks) == 1
    assert b"".join(p for p, _ in fallbacks[0].pushed) == audio
    assert fallbacks[0].pushed[0][1] == pytest.approx(T0)
    assert fallbacks[0].closed
    assert len(notices) == 1 and notices[0]["engine"] == "local"
    assert notices[0]["speaker"]["id"] == "a"


async def test_drop_mid_sentence_keeps_finalized_text_and_replays_the_rest_locally():
    async def script(ws, received):
        if received == 1.0:
            await ws.send(results("first part", 0.0, 1.0, is_final=True))
        elif received == 1.2:
            await ws.send(results("second", 1.0, 0.2, is_final=False))
        elif received == 1.5:
            return "drop"

    # First connection accepted, the reconnect is refused: the stream must go local.
    async with FakeDeepgram(script, accept=1) as server:
        stream, emitted, stored, notices, fallbacks = build(server.url)
        audio = tone(2.0)
        await feed(stream, audio)
        await stream.close()

    assert [u.text for u in stored] == ["first part"]
    assert len(fallbacks) == 1
    replayed = b"".join(p for p, _ in fallbacks[0].pushed)
    # Everything after Deepgram's last confirmed second goes to the local engine, exactly once.
    assert replayed == audio[int(1.0 * SR) * 2 :]
    assert fallbacks[0].pushed[0][1] == pytest.approx(T0 + 1.0, abs=0.01)
    # The bubble shows only the confirmed text; the unconfirmed interim word is not kept.
    last = [e for e in emitted if e["utterance_id"] == stored[0].id][-1]
    assert last["final"] and last["text"] == "first part"
    assert notices and notices[0]["engine"] == "local"


async def test_drop_then_successful_reconnect_stays_on_deepgram():
    sent = {"final": False}

    async def script(ws, received):
        if received == 0.5 and server.connections == 1:
            return "drop"
        if received is not None and server.connections == 2 and received >= 1.0 and not sent["final"]:
            sent["final"] = True
            await ws.send(results("still here", 0.0, 1.0, is_final=True, speech_final=True))

    async with FakeDeepgram(script) as server:
        stream, emitted, stored, notices, fallbacks = build(server.url)
        await feed(stream, tone(2.0))
        await stream.close()

    assert server.connections == 2
    assert not fallbacks and not notices
    assert [u.text for u in stored] == ["still here"]
    # Reconnect replays the unconfirmed audio, so its clock starts at the first sample again.
    assert stored[0].start_ts == pytest.approx(T0 + 0.1, abs=0.01)


async def test_import_transcribe_uses_rest_api():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        body = {"results": {"channels": [{"alternatives": [{"transcript": "namaste dosto"}]}]}}
        return httpx.Response(200, json=body)

    stt = DeepgramSTT(Settings(deepgram_api_key="k"), fallback=RecordingSTT("local"), transport=httpx.MockTransport(handler))
    assert await stt.transcribe(tone(1.0), SR) == "namaste dosto"
    assert seen["auth"] == "Token k"
    assert "model=nova-3" in seen["url"] and "language=multi" in seen["url"]


async def test_import_transcribe_falls_back_when_deepgram_fails():
    fallback = RecordingSTT("local text")
    transport = httpx.MockTransport(lambda request: httpx.Response(402, json={"err_msg": "out of credit"}))
    stt = DeepgramSTT(Settings(deepgram_api_key="k"), fallback=fallback, transport=transport)
    assert await stt.transcribe(tone(1.0), SR) == "local text"
    assert len(fallback.calls) == 1


def test_make_stt_without_key_uses_fallback_directly(monkeypatch):
    import app.stt as stt_module

    stt = stt_module.make_stt(Settings(stt_provider="deepgram", stt_fallback="mock", deepgram_api_key=None))
    assert stt.name == "mock"


def test_make_stt_with_key_wraps_fallback():
    import app.stt as stt_module

    stt = stt_module.make_stt(Settings(stt_provider="deepgram", stt_fallback="mock", deepgram_api_key="k"))
    assert stt.name == "deepgram (fallback: mock)"
    assert stt.fallback.name == "mock"


async def test_socket_handlers_use_deepgram_stream_with_local_fallback():
    import socketio

    from app.llm import LlmChain
    from app.pipeline import ParticipantStream
    from app.sessions import SessionStore
    from app.socket_events import SocketHandlers

    settings = Settings(deepgram_api_key="k", deepgram_url="ws://127.0.0.1:9/v1/listen")
    handlers = SocketHandlers(socketio.AsyncServer(async_mode="asgi"), SessionStore(), settings,
                              DeepgramSTT(settings, fallback=RecordingSTT("local")), LlmChain([]))
    stream = handlers._new_stream("ABCDEF", SPEAKER)
    assert isinstance(stream, DeepgramParticipantStream)

    # A speaker starting a sentence cuts the other speakers' open sentences in the same call only.
    other, elsewhere = handlers._new_stream("ABCDEF", {"id": "b", "name": "Computer"}), handlers._new_stream("ZZZZZZ", {"id": "c", "name": "C"})
    cuts = []
    for s in (other, elsewhere):
        s.cut = lambda ts, s=s: _record(cuts, s, ts)
    await stream._on_start(T0)
    assert cuts == [(other, T0)]
    assert isinstance(stream._make_fallback(), ParticipantStream)


async def _record(cuts, stream, ts):
    cuts.append((stream, ts))


ROOM = {n: {"id": f"u-room{n + 1}", "name": f"Speaker {n + 1}"} for n in range(3)}


async def test_in_room_a_new_voice_ends_the_bubble():
    """One mic, two people: a final result whose words change voice splits into two bubbles, one per speaker."""
    async def script(ws, received):
        if received == 2.0:
            await ws.send(
                results("thanks for coming in. thank you", 0.0, 2.0, is_final=True, speakers=[0, 0, 0, 0, 1, 1])
            )

    async with FakeDeepgram(script) as server:
        stream, emitted, stored, *_ = build(server.url, speaker_for=ROOM.__getitem__)
        await feed(stream, tone(2.0))
        await stream.close()

    assert server.params["diarize"] == "true"
    assert [(u.speaker_name, u.text) for u in stored] == [("Speaker 1", "thanks for coming in."), ("Speaker 2", "thank you")]
    assert stored[0].speaker_id == "u-room1" and stored[1].speaker_id == "u-room2"
    finals = [e for e in emitted if e["final"]]
    assert [f["speaker"]["name"] for f in finals] == ["Speaker 1", "Speaker 2"]


async def test_in_room_a_voice_change_in_live_text_waits_for_the_final():
    """Live labels flip often: live text of another voice does not end a bubble that has final words."""
    async def script(ws, received):
        if received == 1.0:
            await ws.send(results("hello there", 0.0, 1.0, is_final=True, speakers=[0, 0]))
        elif received == 1.5:
            await ws.send(results("my friend", 1.0, 0.5, is_final=False, speakers=[1, 1]))
        elif received == 2.0:
            await ws.send(results("my friend", 1.0, 1.0, is_final=True, speakers=[0, 0]))

    async with FakeDeepgram(script) as server:
        stream, emitted, stored, *_ = build(server.url, speaker_for=ROOM.__getitem__)
        await feed(stream, tone(2.0))
        await stream.close()

    assert [(u.speaker_name, u.text) for u in stored] == [("Speaker 1", "hello there my friend")]
    assert {e["speaker"]["name"] for e in emitted} == {"Speaker 1"}


async def test_without_diarization_the_url_has_no_diarize():
    async def script(ws, received):
        return None

    async with FakeDeepgram(script) as server:
        stream, *_ = build(server.url)
        await feed(stream, tone(0.2))
        await stream.close()

    assert "diarize" not in server.params


async def test_diarize_asks_for_speakers_and_returns_words_with_them():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        words = [
            {"word": "six", "punctuated_word": "Six", "start": 0.1, "end": 0.4, "speaker": 0},
            {"word": "okay", "start": 1.0, "end": 1.3, "speaker": 1},
        ]
        return httpx.Response(200, json={"results": {"channels": [{"alternatives": [{"words": words}]}]}})

    settings = Settings(deepgram_api_key="k", deepgram_language="en")
    stt = DeepgramSTT(settings, fallback=RecordingSTT(), transport=httpx.MockTransport(handler))
    words = await stt.diarize(silence(1.0), SR)

    assert seen["params"]["diarize"] == "true" and seen["params"]["model"] == "nova-3"
    assert words == [
        {"start": 0.1, "end": 0.4, "word": "Six", "speaker": 0},
        {"start": 1.0, "end": 1.3, "word": "okay", "speaker": 1},
    ]
