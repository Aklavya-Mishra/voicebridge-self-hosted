"""One live speaker's path: audio -> segmenter -> STT -> transcript events (or audio -> streaming STT)."""
import asyncio
import json
import logging
import time
import uuid
from typing import Awaitable, Callable

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from app.segmenter import CANCEL, PARTIAL, Segment, Segmenter
from app.sessions import Utterance
from app.stt import STTProvider, clean_transcript

logger = logging.getLogger("voicebridge")

Emit = Callable[[dict], Awaitable[None]]
OnFinal = Callable[[Utterance], Awaitable[None]]


class ParticipantStream:
    def __init__(
        self,
        *,
        code: str,
        speaker: dict,
        segmenter: Segmenter,
        stt: STTProvider,
        sample_rate: int,
        language: str | None,
        semaphore: asyncio.Semaphore,
        emit: Emit,
        on_final: OnFinal,
    ):
        self._code = code
        self._speaker = speaker
        self._segmenter = segmenter
        self._stt = stt
        self._sample_rate = sample_rate
        self._language = language
        self._semaphore = semaphore
        self._emit = emit
        self._on_final = on_final
        self._partial_busy: set[str] = set()
        self._finalized: set[str] = set()
        self._tasks: set[asyncio.Task] = set()

    async def push(self, pcm: bytes, ts: float) -> None:
        for segment in self._segmenter.feed(pcm, ts):
            await self._dispatch(segment)

    async def close(self) -> None:
        for segment in self._segmenter.flush():
            await self._dispatch(segment)
        if self._tasks:
            # Shielded: if the caller is cancelled (client disconnected right after stopping),
            # the pending transcriptions still finish, get stored and reach the other participant.
            await asyncio.shield(asyncio.gather(*self._tasks, return_exceptions=True))

    async def _dispatch(self, segment: Segment) -> None:
        if segment.kind == PARTIAL:
            # Latest-wins: skip a partial while the previous one for this utterance is still running.
            if segment.utterance_id in self._partial_busy:
                return
            self._partial_busy.add(segment.utterance_id)
            self._spawn(self._run_partial(segment))
        elif segment.kind == CANCEL:
            self._finalized.add(segment.utterance_id)
            await self._emit(self._payload(segment, "", final=True))
        else:
            self._spawn(self._run_final(segment))

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _transcribe(self, pcm: bytes) -> str:
        async with self._semaphore:
            try:
                text = await self._stt.transcribe(pcm, self._sample_rate, self._language)
            except Exception:
                logger.exception("STT failed for session %s", self._code)
                return ""
        return clean_transcript(text)

    async def _run_partial(self, segment: Segment) -> None:
        try:
            text = await self._transcribe(segment.pcm)
        finally:
            self._partial_busy.discard(segment.utterance_id)
        if text and segment.utterance_id not in self._finalized:
            await self._emit(self._payload(segment, text, final=False))

    async def _run_final(self, segment: Segment) -> None:
        text = await self._transcribe(segment.pcm)
        self._finalized.add(segment.utterance_id)
        if text:
            await self._on_final(
                Utterance(
                    id=segment.utterance_id,
                    speaker_id=self._speaker["id"],
                    speaker_name=self._speaker["name"],
                    text=text,
                    start_ts=segment.start_ts,
                    end_ts=segment.end_ts,
                )
            )
        await self._emit(self._payload(segment, text, final=True))

    def _payload(self, segment: Segment, text: str, final: bool) -> dict:
        return {
            "code": self._code,
            "utterance_id": segment.utterance_id,
            "speaker": {"id": self._speaker["id"], "name": self._speaker["name"]},
            "text": text,
            "start_ts": segment.start_ts,
            "end_ts": segment.end_ts,
            "final": final,
        }


# Clip sent to the refiner: from shortly before the first word to shortly after the last one.
# Token timestamps trail the real word onset, and the model reports a sentence end only after the
# next word may have started, so the clip is cut by timestamps, never at the end-of-sentence signal.
REFINE_LEAD_S = 0.5
REFINE_TAIL_S = 0.35
# Recent audio kept for the refiner, across sentence boundaries.
AUDIO_BUFFER_S = 30.0
# Re-anchor the stream clock when chunk timestamps drift further than this (seconds).
CLOCK_RESYNC_S = 1.0


class StreamingParticipantStream:
    """One live speaker on a streaming model (sherpa): words are emitted as they are recognised.

    The model marks the end of each sentence. The sentence is then optionally re-transcribed by
    the refiner (Whisper) for accuracy before it is stored as final.
    Times below are seconds of audio fed since the stream opened unless named *_ts (server clock).
    """

    def __init__(
        self,
        *,
        code: str,
        speaker: dict,
        stt,  # SherpaSTT
        sample_rate: int,
        language: str | None,
        semaphore: asyncio.Semaphore,
        emit: Emit,
        on_final: OnFinal,
    ):
        self._code = code
        self._speaker = speaker
        self._stt = stt
        self._sample_rate = sample_rate
        self._language = language
        self._semaphore = semaphore
        self._emit = emit
        self._on_final = on_final
        self._stream = stt.open_stream()
        # Chunks arrive as separate tasks; the lock keeps them in order and off the model concurrently.
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._closed = False
        self._fed = 0.0
        self._clock_base_ts: float | None = None  # server time of audio second 0
        self._buffer = bytearray()
        self._buffer_start = 0.0
        self._prev_end = 0.0  # end of the previous sentence's refiner clip
        self._reset_at = 0.0  # where the model's timestamps start counting
        self._new_sentence()

    def _new_sentence(self) -> None:
        self._utterance_id: str | None = None
        self._first_word = 0.0
        self._last_word = 0.0
        self._text = ""

    async def push(self, pcm: bytes, ts: float) -> None:
        async with self._lock:
            if self._closed:
                return
            if self._clock_base_ts is None or abs(ts - self._fed - self._clock_base_ts) > CLOCK_RESYNC_S:
                self._clock_base_ts = ts - self._fed
            self._buffer += pcm
            self._fed += len(pcm) / 2 / self._sample_rate
            excess = len(self._buffer) - int(AUDIO_BUFFER_S * self._sample_rate) * 2
            if excess > 0:
                del self._buffer[:excess]
                self._buffer_start += excess / 2 / self._sample_rate
            result = await asyncio.to_thread(self._stt.decode, self._stream, pcm)
            await self._handle(result)
            if result.endpoint:
                if self._utterance_id:
                    logger.debug("Sentence end detected %.2fs after its last word", self._fed - self._last_word)
                self._stt.reset(self._stream)
                self._reset_at = self._fed
                self._new_sentence()

    async def close(self) -> None:
        async with self._lock:
            if not self._closed:
                self._closed = True
                if self._clock_base_ts is not None:
                    await self._handle(await asyncio.to_thread(self._stt.finish, self._stream))
        if self._tasks:
            # Shielded like ParticipantStream.close: pending refinements still finish and get stored.
            await asyncio.shield(asyncio.gather(*self._tasks, return_exceptions=True))

    async def _handle(self, result) -> None:
        if result.text and result.timestamps:
            if self._utterance_id is None:
                self._utterance_id = uuid.uuid4().hex[:12]
                self._first_word = self._reset_at + result.timestamps[0]
            self._last_word = self._reset_at + result.timestamps[-1]
        if self._utterance_id is None:
            return
        if result.text and result.text != self._text:
            self._text = result.text
            await self._emit(self._partial_payload(result.text))
        if result.endpoint:
            await self._finish_sentence()

    async def _finish_sentence(self) -> None:
        clip_start = max(self._prev_end, self._first_word - REFINE_LEAD_S, self._buffer_start)
        clip_end = min(self._fed, self._last_word + REFINE_TAIL_S)
        self._prev_end = clip_end
        utterance = Utterance(
            id=self._utterance_id,
            speaker_id=self._speaker["id"],
            speaker_name=self._speaker["name"],
            text=self._text,
            start_ts=self._clock_base_ts + self._first_word,
            end_ts=self._clock_base_ts + clip_end,
        )
        refiner = getattr(self._stt, "refiner", None)
        if refiner is None:
            await self._store(utterance)
            return
        pcm = bytes(self._buffer[self._byte_at(clip_start) : self._byte_at(clip_end)])
        task = asyncio.create_task(self._refine(refiner, utterance, pcm))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _byte_at(self, seconds: float) -> int:
        return max(0, int((seconds - self._buffer_start) * self._sample_rate)) * 2

    async def _refine(self, refiner: STTProvider, utterance: Utterance, pcm: bytes) -> None:
        queued = time.perf_counter()
        async with self._semaphore:
            started = time.perf_counter()
            try:
                text = clean_transcript(await refiner.transcribe(pcm, self._sample_rate, self._language))
            except Exception:
                logger.exception("Refinement failed for session %s; keeping the streaming text", self._code)
                text = ""
        logger.debug(
            "Refined %.1fs clip in %.2fs (queued %.2fs)",
            len(pcm) / 2 / self._sample_rate, time.perf_counter() - started, started - queued,
        )
        if text:
            utterance.text = text
        await self._store(utterance)

    async def _store(self, utterance: Utterance) -> None:
        await self._on_final(utterance)
        await self._emit(
            {
                "code": self._code,
                "utterance_id": utterance.id,
                "speaker": {"id": self._speaker["id"], "name": self._speaker["name"]},
                "text": utterance.text,
                "start_ts": utterance.start_ts,
                "end_ts": utterance.end_ts,
                "final": True,
            }
        )

    def _partial_payload(self, text: str) -> dict:
        return {
            "code": self._code,
            "utterance_id": self._utterance_id,
            "speaker": {"id": self._speaker["id"], "name": self._speaker["name"]},
            "text": text,
            "start_ts": self._clock_base_ts + self._first_word,
            "end_ts": self._clock_base_ts + self._fed,
            "final": False,
        }


# Idle connections are closed by Deepgram after ~10 s without data; ping well before that.
KEEPALIVE_S = 4.0
# How long close() waits for Deepgram's last results after asking it to flush.
CLOSE_WAIT_S = 5.0
REPLAY_CHUNK_S = 0.1


class DeepgramParticipantStream:
    """One live speaker streamed to Deepgram, with a local stream as fallback.

    Deepgram sends interim results (shown as live text) and final results for stretches of audio;
    a pause (speech_final / UtteranceEnd) ends the sentence, which is then stored.
    Audio Deepgram has not finalised yet is kept. If the connection drops, one reconnect is tried
    and that audio is re-sent; if Deepgram is unreachable, the audio is replayed into the local
    stream from make_fallback() and this speaker stays local until the mic is restarted.
    Times below are seconds of audio received since the stream opened unless named *_ts.
    """

    def __init__(
        self,
        *,
        code: str,
        speaker: dict,
        stt,  # DeepgramSTT
        sample_rate: int,
        emit: Emit,
        on_final: OnFinal,
        notify: Emit,
        make_fallback: Callable[[], "ParticipantStream | StreamingParticipantStream"],
    ):
        self._code = code
        self._speaker = speaker
        self._stt = stt
        self._sample_rate = sample_rate
        self._emit = emit
        self._on_final = on_final
        self._notify = notify
        self._make_fallback = make_fallback
        self._lock = asyncio.Lock()
        self._conn = None
        self._receiver: asyncio.Task | None = None
        self._keepalive: asyncio.Task | None = None
        self._dropped = False
        self._closing = False
        self._local = None
        self._fed = 0.0
        self._last_send = 0.0
        self._clock_base_ts: float | None = None  # server time of audio second 0
        self._buffer = bytearray()  # audio Deepgram has not finalised yet
        self._buffer_start = 0.0  # = end of Deepgram's last final result
        self._conn_offset = 0.0  # Deepgram's times on this connection count from here
        self._new_sentence()

    def _new_sentence(self) -> None:
        self._utterance_id: str | None = None
        self._parts: list[str] = []
        self._first_word = 0.0
        self._last_word = 0.0

    async def push(self, pcm: bytes, ts: float) -> None:
        async with self._lock:
            if self._closing:
                return
            if self._local:
                await self._local.push(pcm, ts)
                return
            if self._clock_base_ts is None or abs(ts - self._fed - self._clock_base_ts) > CLOCK_RESYNC_S:
                self._clock_base_ts = ts - self._fed
            self._buffer += pcm
            self._fed += len(pcm) / 2 / self._sample_rate
            if self._conn is not None and not self._dropped:
                try:
                    await self._conn.send(pcm)
                    self._last_send = time.monotonic()
                    return
                except ConnectionClosed:
                    self._dropped = True
            if self._dropped:
                logger.warning("Deepgram connection dropped for session %s; reconnecting", self._code)
            if not await self._connect():
                await self._go_local("Deepgram is unreachable")

    async def close(self) -> None:
        async with self._lock:
            if self._closing:
                return
            self._closing = True
            if self._local is None and self._conn is not None:
                if not self._dropped:
                    try:
                        await self._conn.send(json.dumps({"type": "CloseStream"}))
                        await asyncio.wait_for(asyncio.shield(self._receiver), CLOSE_WAIT_S)
                    except (ConnectionClosed, asyncio.TimeoutError):
                        pass
                await self._disconnect()
                if self._dropped and self._buffer:
                    await self._go_local("Deepgram connection dropped")
            if self._local is None:
                await self._finish_sentence()
        if self._local is not None:
            await self._local.close()

    async def _connect(self) -> bool:
        await self._disconnect()
        try:
            self._conn = await asyncio.wait_for(
                ws_connect(self._stt.live_url(), additional_headers=self._stt.headers),
                self._stt.connect_timeout_s,
            )
        except Exception as exc:
            logger.warning("Cannot connect to Deepgram for session %s: %s", self._code, exc)
            self._conn = None
            return False
        self._dropped = False
        self._conn_offset = self._buffer_start
        conn = self._conn
        self._receiver = asyncio.create_task(self._receive(conn))
        self._keepalive = asyncio.create_task(self._keep_alive(conn))
        try:
            step = int(REPLAY_CHUNK_S * self._sample_rate) * 2
            for i in range(0, len(self._buffer), step):
                await conn.send(bytes(self._buffer[i : i + step]))
            self._last_send = time.monotonic()
        except ConnectionClosed:
            return False
        return True

    async def _disconnect(self) -> None:
        for task in (self._keepalive, self._receiver):
            if task and not task.done():
                task.cancel()
        if self._conn is not None:
            await self._conn.close()
        self._conn = self._receiver = self._keepalive = None

    async def _go_local(self, reason: str) -> None:
        """Switch to the local engine: keep what Deepgram finalised, replay the rest locally."""
        await self._disconnect()
        await self._finish_sentence()
        self._local = self._make_fallback()
        logger.warning("%s for session %s; using local transcription", reason, self._code)
        await self._notify({"code": self._code, "speaker": self._speaker_ref(), "engine": "local", "reason": reason})
        step = int(REPLAY_CHUNK_S * self._sample_rate) * 2
        for i in range(0, len(self._buffer), step):
            ts = self._clock_base_ts + self._buffer_start + i / 2 / self._sample_rate
            await self._local.push(bytes(self._buffer[i : i + step]), ts)
        self._buffer.clear()

    async def _keep_alive(self, conn) -> None:
        try:
            while True:
                await asyncio.sleep(KEEPALIVE_S / 2)
                if time.monotonic() - self._last_send >= KEEPALIVE_S:
                    await conn.send(json.dumps({"type": "KeepAlive"}))
        except ConnectionClosed:
            pass

    async def _receive(self, conn) -> None:
        try:
            async for message in conn:
                await self._handle(json.loads(message))
        except ConnectionClosed:
            pass
        except Exception:
            logger.exception("Bad message from Deepgram for session %s", self._code)
        finally:
            if not self._closing:
                self._dropped = True

    async def _handle(self, message: dict) -> None:
        if message.get("type") == "UtteranceEnd":
            await self._finish_sentence()
            return
        if message.get("type") != "Results":
            return
        alternative = message["channel"]["alternatives"][0]
        text = alternative.get("transcript", "").strip()
        words = alternative.get("words") or []
        if text and words and self._utterance_id is None:
            self._utterance_id = uuid.uuid4().hex[:12]
            self._first_word = self._conn_offset + words[0]["start"]
        if message.get("is_final"):
            if text and words:
                self._parts.append(text)
                self._last_word = self._conn_offset + words[-1]["end"]
            self._confirm(self._conn_offset + message["start"] + message["duration"])
            if message.get("speech_final"):
                await self._finish_sentence()
            elif text:
                await self._emit(self._partial_payload(" ".join(self._parts)))
        elif text:
            await self._emit(self._partial_payload(" ".join([*self._parts, text])))

    def _confirm(self, until: float) -> None:
        """Deepgram finalised audio up to `until`: it no longer needs to be kept for a replay."""
        drop = int((until - self._buffer_start) * self._sample_rate) * 2
        if drop > 0:
            del self._buffer[:drop]
            self._buffer_start += drop / 2 / self._sample_rate

    async def _finish_sentence(self) -> None:
        if self._utterance_id is None:
            return
        utterance_id, parts, first, last = self._utterance_id, self._parts, self._first_word, self._last_word
        self._new_sentence()
        if not parts:
            # Only interim text was shown; a final with empty text removes that bubble.
            await self._emit(self._payload(utterance_id, "", first, self._fed, final=True))
            return
        utterance = Utterance(
            id=utterance_id,
            speaker_id=self._speaker["id"],
            speaker_name=self._speaker["name"],
            text=clean_transcript(" ".join(parts)),
            start_ts=self._clock_base_ts + first,
            end_ts=self._clock_base_ts + last,
        )
        await self._on_final(utterance)
        await self._emit(self._payload(utterance_id, utterance.text, first, last, final=True))

    def _speaker_ref(self) -> dict:
        return {"id": self._speaker["id"], "name": self._speaker["name"]}

    def _partial_payload(self, text: str) -> dict:
        return self._payload(self._utterance_id, text, self._first_word, self._fed, final=False)

    def _payload(self, utterance_id: str, text: str, start: float, end: float, final: bool) -> dict:
        return {
            "code": self._code,
            "utterance_id": utterance_id,
            "speaker": self._speaker_ref(),
            "text": text,
            "start_ts": self._clock_base_ts + start,
            "end_ts": self._clock_base_ts + end,
            "final": final,
        }
