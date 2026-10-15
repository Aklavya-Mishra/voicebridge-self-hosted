"""One live speaker's path: audio -> segmenter -> STT -> transcript events (or audio -> Deepgram)."""
import asyncio
import json
import logging
import time
import uuid
from typing import Awaitable, Callable

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from app.analysis import mask_text
from app.segmenter import CANCEL, PARTIAL, Segment, Segmenter
from app.sessions import Utterance
from app.stt import STTProvider, clean_transcript

logger = logging.getLogger("voicebridge")

Emit = Callable[[dict], Awaitable[None]]
OnFinal = Callable[[Utterance], Awaitable[None]]
# (start, end, text, speaker) of one recognised word; speaker is Deepgram's voice number (0 without diarization).
Word = tuple[float, float, str, int]
# One Deepgram result: its words, and its text as Deepgram formatted it ("87548734", not "eight seven...").
Part = tuple[list[Word], str]


def _words_text(words: list[Word]) -> str:
    return " ".join(w[2] for w in words)


def _text(parts: list[Part]) -> str:
    return " ".join(text for _, text in parts)


def _split(parts: list[Part], at: float) -> tuple[list[Part], list[Part]]:
    """Parts with words starting before `at`, and from `at` on. A part spanning `at` is split by its words."""
    before: list[Part] = []
    after: list[Part] = []
    for words, text in parts:
        early = [w for w in words if w[0] < at]
        late = words[len(early):]
        if not late:
            before.append((words, text))
        elif not early:
            after.append((words, text))
        else:
            before.append((early, _words_text(early)))
            after.append((late, _words_text(late)))
    return before, after


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
        voices=None,  # in-room: a VoiceTracker (app.stt.sherpa) giving each line's voice number
        speaker_for: Callable[[int], dict] | None = None,  # in-room: voice number -> speaker
    ):
        self._code = code
        self._speaker = speaker  # in-room: the speaker of the latest line, shown on live text
        self._voices = voices
        self._speaker_for = speaker_for
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

    async def _voice_of(self, pcm: bytes) -> dict:
        """In-room: who said this line, by voice print. Otherwise the stream's one speaker."""
        if self._voices is None or self._speaker_for is None:
            return self._speaker
        try:
            label = await asyncio.to_thread(self._voices.label, pcm)
        except Exception:
            logger.exception("Voice print failed for session %s", self._code)
            return self._speaker
        self._speaker = self._speaker_for(label)
        return self._speaker

    async def _run_final(self, segment: Segment) -> None:
        text, speaker = await asyncio.gather(self._transcribe(segment.pcm), self._voice_of(segment.pcm))
        self._finalized.add(segment.utterance_id)
        if text:
            await self._on_final(
                Utterance(
                    id=segment.utterance_id,
                    speaker_id=speaker["id"],
                    speaker_name=speaker["name"],
                    text=text,
                    start_ts=segment.start_ts,
                    end_ts=segment.end_ts,
                )
            )
        await self._emit(self._payload(segment, text, final=True, speaker=speaker))

    def _payload(self, segment: Segment, text: str, final: bool, speaker: dict | None = None) -> dict:
        speaker = speaker or self._speaker
        return {
            "code": self._code,
            "utterance_id": segment.utterance_id,
            "speaker": {"id": speaker["id"], "name": speaker["name"]},
            "text": text,
            "start_ts": segment.start_ts,
            "end_ts": segment.end_ts,
            "final": final,
        }


# Re-anchor the stream clock when chunk timestamps drift further than this (seconds).
CLOCK_RESYNC_S = 1.0


# Idle connections are closed by Deepgram after ~10 s without data; ping well before that.
KEEPALIVE_S = 4.0
# How long close() waits for Deepgram's last results after asking it to flush.
CLOSE_WAIT_S = 5.0
REPLAY_CHUNK_S = 0.1


class DeepgramParticipantStream:
    """One live speaker streamed to Deepgram, with a local stream as fallback.

    Deepgram sends interim results (shown as live text) and final results for stretches of audio. All of
    them go into one bubble until the speaker is silent for `bubble_gap_s` (3 s): then it is stored. Short
    pauses (Deepgram's speech_final) do not end the bubble.
    Audio Deepgram has not finalised yet is kept. If the connection drops, one reconnect is tried
    and that audio is re-sent; if Deepgram is unreachable, the audio is replayed into the local
    stream from make_fallback() and this speaker stays local until the mic is restarted.
    When another speaker in the call starts a sentence, cut() ends this speaker's open sentence at that
    moment and the words after it continue in a new bubble, so the chat reads in speaking order.
    With `speaker_for` (in-room mode: one mic, several people), Deepgram numbers the voices it hears and each
    word goes to speaker_for(number); a bubble also ends when a final result says another voice is speaking.
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
        make_fallback: Callable[[], ParticipantStream],
        on_start: Callable[[float], Awaitable[None]] | None = None,
        bubble_gap_s: float = 3.0,
        speaker_for: Callable[[int], dict] | None = None,
    ):
        self._code = code
        self._speaker_for = speaker_for
        self._sentence_speaker = speaker  # who the open bubble belongs to
        self._bubble_gap_s = bubble_gap_s
        self._speaker = speaker
        self._stt = stt
        self._sample_rate = sample_rate
        self._emit = emit
        self._on_final = on_final
        self._notify = notify
        self._make_fallback = make_fallback
        self._on_start = on_start  # called with the server time of each new sentence's first word
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
        self._interim: list[Part] = []  # the latest live result, not final yet
        self._floor = 0.0  # words starting before this were already stored by cut()
        self._new_sentence()

    def _new_sentence(self) -> None:
        self._utterance_id: str | None = None
        self._parts: list[Part] = []  # finalised results of the open sentence
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
                ws_connect(self._stt.live_url(diarize=self._speaker_for is not None), additional_headers=self._stt.headers),
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
        if message.get("type") != "Results":  # UtteranceEnd too: only a 3 s silence ends a bubble
            return
        alternative = message["channel"]["alternatives"][0]
        all_words = [
            (
                self._conn_offset + w["start"],
                self._conn_offset + w["end"],
                w.get("punctuated_word") or w["word"],
                w.get("speaker", 0),
            )
            for w in alternative.get("words") or []
        ]
        words = [w for w in all_words if w[0] >= self._floor]
        text = alternative.get("transcript", "").strip() if len(words) == len(all_words) else _words_text(words)
        part = (words, text) if words and text else None
        # Silence is measured on Deepgram's clock (audio it has answered for), not on the audio sent:
        # Deepgram answers 2-3 s late, so words after a short pause may simply not be back yet.
        heard_until = self._conn_offset + message["start"] + message["duration"]
        if not message.get("is_final"):
            self._interim = [part] if part else []
            if part:
                if not self._parts:  # live labels are unsure: only a final result ends a bubble with text
                    await self._turn_to(words[0][3])
                await self._start_sentence(words[0][0])
                await self._emit(self._partial_payload())
            else:
                await self._end_if_silent(heard_until)
            return
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Deepgram final (speech_final=%s, %d words) for session %s: %r",
                message.get("speech_final"), len(words), self._code, mask_text(text),
            )
        if part:
            for run in self._runs(self._with_dropped_words(part)):
                await self._turn_to(run[0][0][3])
                await self._start_sentence(run[0][0][0])
                self._parts.append(run)
                self._last_word = run[0][-1][1]
        self._interim = []
        self._confirm(heard_until)
        if part:
            await self._emit(self._partial_payload())
        else:
            await self._end_if_silent(heard_until)

    def _runs(self, part: Part) -> list[Part]:
        """The part split where the speaking voice changes (one run without diarization)."""
        words, text = part
        if self._speaker_for is None or len({w[3] for w in words}) == 1:
            return [part]
        runs: list[list[Word]] = []
        for w in words:
            if runs and runs[-1][-1][3] == w[3]:
                runs[-1].append(w)
            else:
                runs.append([w])
        return [(run, _words_text(run)) for run in runs]

    async def _turn_to(self, label: int) -> None:
        """In-room mode: the next words are voice `label`. Another voice than the open bubble's ends it."""
        if self._speaker_for is None:
            return
        speaker = self._speaker_for(label)
        if self._utterance_id is not None and speaker["id"] != self._sentence_speaker["id"]:
            if self._parts:
                await self._finish_sentence()
            # else only live text was shown: that bubble just changes speaker
        self._sentence_speaker = speaker

    async def _end_if_silent(self, heard_until: float) -> None:
        """Deepgram heard nothing from the last word up to `heard_until`: a long enough silence ends the bubble."""
        if self._parts and not self._interim and heard_until - self._last_word >= self._bubble_gap_s:
            await self._finish_sentence()

    async def _start_sentence(self, first_word: float) -> None:
        if self._utterance_id is not None:
            return
        self._utterance_id = uuid.uuid4().hex[:12]
        self._first_word = first_word
        if self._on_start:
            await self._on_start(self._clock_base_ts + first_word)

    def _with_dropped_words(self, part: Part) -> Part:
        """Deepgram's final sometimes leaves out words its live results showed, e.g. only the number of
        "my card number is 8754...". Words the live result had before the final's first word are kept."""
        words, text = part
        dropped = [w for interim_words, _ in self._interim for w in interim_words if w[1] <= words[0][0]]
        if not dropped:
            return part
        logger.debug("Kept %d words missing from Deepgram's final for session %s", len(dropped), self._code)
        return dropped + words, f"{_words_text(dropped)} {text}"

    async def cut(self, at_ts: float) -> None:
        """Another speaker started at `at_ts` (server clock): store this speaker's words spoken before then
        as a finished bubble; later words continue in a new one, shown after the other speaker's."""
        if self._local or self._utterance_id is None or self._clock_base_ts is None:
            return
        at = at_ts - self._clock_base_ts
        spoken, rest = _split(self._parts, at)
        spoken_interim, rest_interim = _split(self._interim, at)
        spoken += spoken_interim
        if at <= self._first_word or not spoken:
            return
        utterance_id, first = self._utterance_id, self._first_word
        self._floor = max(self._floor, at)
        self._new_sentence()
        self._interim = rest_interim
        if rest or rest_interim:
            # The continuation is not a new sentence of this speaker, so it does not cut anyone.
            self._utterance_id = uuid.uuid4().hex[:12]
            self._first_word = (rest or rest_interim)[0][0][0][0]
            self._parts = rest
            self._last_word = rest[-1][0][-1][1] if rest else 0.0
        await self._store(utterance_id, spoken, first, spoken[-1][0][-1][1])
        if self._utterance_id:
            await self._emit(self._partial_payload())

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
        await self._store(utterance_id, parts, first, last)

    async def _store(self, utterance_id: str, parts: list[Part], first: float, last: float) -> None:
        utterance = Utterance(
            id=utterance_id,
            speaker_id=self._sentence_speaker["id"],
            speaker_name=self._sentence_speaker["name"],
            text=clean_transcript(_text(parts)),
            start_ts=self._clock_base_ts + first,
            end_ts=self._clock_base_ts + last,
        )
        await self._on_final(utterance)
        await self._emit(self._payload(utterance_id, utterance.text, first, last, final=True))

    def _speaker_ref(self) -> dict:
        return {"id": self._speaker["id"], "name": self._speaker["name"]}

    def _partial_payload(self) -> dict:
        text = _text(self._parts + self._interim)
        return self._payload(self._utterance_id, text, self._first_word, self._fed, final=False)

    def _payload(self, utterance_id: str, text: str, start: float, end: float, final: bool) -> dict:
        return {
            "code": self._code,
            "utterance_id": utterance_id,
            "speaker": {"id": self._sentence_speaker["id"], "name": self._sentence_speaker["name"]},
            "text": text,
            "start_ts": self._clock_base_ts + start,
            "end_ts": self._clock_base_ts + end,
            "final": final,
        }
