"""Socket.IO events for live sessions.

Client -> server: join_session {code, name?}, rename_speaker {code, speaker_id, name} -> ack {ok, error?},
  audio_chunk {code, audio, channels?}, stop_audio {code}, leave_session {code}
  audio is PCM16 at the configured sample rate. channels=2 means interleaved stereo from one endpoint:
  left = the user's mic, right = the call audio (the remote person, who does not run VoiceBridge).
  Each channel gets its own STT stream; the channel decides the speaker.

Desktop agent (connects with auth {token, client: "agent"} using the web UI's token, handed over locally;
one per user):
  web -> server: agent_start {code} -> ack {ok, error?}; agent_stop {code} -> ack {ok}
  web -> server: mute_mic {code, muted} -> ack {ok, error?}: while muted the agent's mic channel is dropped
  (what was said before is saved); the call audio keeps being transcribed.
  server -> agent: start_recording {code} -> ack {ok, error?}; stop_recording {code}
  agent -> server: agent_ping {code|null} every few seconds and on each (re)connect: re-registers the agent and
  the call it records after a reconnect.
  server -> web: agent_status {online}
  The agent streams stereo audio_chunk (mic + speaker loopback) only into sessions it was started for.
web -> server: set_mode {code, mode: call|room} -> ack {ok, error?} before recording; server -> room:
  mode_changed {code, mode}.
In-room sessions (mode "room"): one mic hears everyone. Only the mic is used (mono, or the left channel of the agent's
  stereo); Deepgram, or sherpa's voice prints, tell the voices apart and lines go to "<user id>-room<n>" ("Speaker n"
  until renamed). When the mic stops, the meeting audio is checked again as a whole and the transcript of that span
  is rebuilt (app.room).
Server -> client: participants {code, participants}, transcript {...see pipeline payload},
                  transcript_reset {code, transcript} when an in-room meeting's transcript was rebuilt,
                  stt_status {code, speaker, engine, reason} when a speaker falls back to local STT,
                  speaker_renamed {code, speaker_id, name},
                  live hints (app.nba): nba_hint {code, id, text, kind, intent, priority, ts},
                  nba_facts {code, facts}, nba_domain {code, domain}, nba_status {code, state, reason?},
                  Jira tickets (app.tickets): ticket_draft {code, id, status, title, description, requested_by,
                  trigger, ts, key?, url?, error?}
Web -> server (Jira tickets): ticket_create {code, id, title, description} -> ack {ok, ticket?, error?},
                  ticket_dismiss {code, id} -> ack {ok, ticket?, error?}
"""
import asyncio
import logging
import time

import numpy as np
import socketio

from app.analysis import analyse_session, mask_text
from app.auth import AuthError, verify_token
from app.config import Settings
from app.llm import LlmChain
from app.nba.service import NbaService
from app.pipeline import DeepgramParticipantStream, ParticipantStream
from app.room import RoomRecorder, recheck
from app.segmenter import Segmenter
from app.sessions import (
    CALL_SPEAKER_NAME,
    SessionError,
    SessionStore,
    Utterance,
    call_speaker_id,
    normalize_code,
    room_speaker_id,
    room_speaker_name,
)
from app.stt import STTProvider
from app.stt.deepgram import DeepgramSTT
from app.tickets import TicketService
from app.vad import make_vad

logger = logging.getLogger("voicebridge")

LiveStream = ParticipantStream | DeepgramParticipantStream

# How long the agent has to confirm start_recording.
AGENT_ACK_TIMEOUT_S = 5.0

# After the last stream of a call closes, wait this long before analysing it (the mic may restart).
ANALYSIS_DELAY_S = 1.0
# In-room with a local engine: pause (ms) that ends a line, so each line holds one voice for its voice print.
ROOM_END_SILENCE_MS = 1000

# Re-anchor a stream's clock to arrival time when the gap is bigger than this (seconds).
CLOCK_RESYNC_S = 1.0


def room_for(code: str) -> str:
    return f"session:{code}"


def user_room(user_id: str) -> str:
    return f"user:{user_id}"


def split_stereo(pcm: bytes) -> tuple[bytes, bytes]:
    frames = np.frombuffer(pcm, dtype=np.int16).reshape(-1, 2)
    return frames[:, 0].tobytes(), frames[:, 1].tobytes()


class SocketHandlers:
    def __init__(
        self,
        sio: socketio.AsyncServer,
        store: SessionStore,
        settings: Settings,
        stt: STTProvider,
        llm: LlmChain,
        nba: NbaService | None = None,
        tickets: TicketService | None = None,
    ):
        self._sio = sio
        self._nba = nba
        self._tickets = tickets
        self._llm = llm
        self._analysis: dict[str, asyncio.Task] = {}  # session code -> pending end-of-call analysis
        self._store = store
        self._settings = settings
        self._stt = stt
        self._semaphore = asyncio.Semaphore(settings.stt_concurrency)
        self._streams: dict[tuple[str, str], LiveStream] = {}
        self._clocks: dict[tuple[str, str], float] = {}
        self._agents: dict[str, str] = {}  # user id -> agent sid
        self._agent_codes: dict[str, set[str]] = {}  # user id -> sessions the agent is recording
        # The speaker dict each live stream reads its name from; a rename updates it in place.
        self._stream_speakers: dict[tuple[str, str], dict] = {}
        self._muted: set[tuple[str, str]] = set()  # (session code, user id) whose mic is muted
        self._closing: set[asyncio.Task] = set()  # streams being flushed in the background
        self._room_audio = RoomRecorder(settings.sample_rate)  # in-room meetings' audio until re-checked
        self._room_user: dict[str, str] = {}  # in-room session code -> user whose mic records it
        events = ("connect", "disconnect", "join_session", "audio_chunk", "stop_audio", "leave_session",
                  "agent_start", "agent_stop", "agent_ping", "rename_speaker", "mute_mic", "set_mode",
                  "ticket_create", "ticket_dismiss")
        for name in events:
            sio.on(name, getattr(self, name))

    async def connect(self, sid: str, environ: dict, auth: dict | None = None) -> None:
        try:
            user = verify_token(self._settings, (auth or {}).get("token"))
        except AuthError as exc:
            raise socketio.exceptions.ConnectionRefusedError("unauthorized") from exc
        is_agent = (auth or {}).get("client") == "agent"
        await self._sio.save_session(sid, {"user": user, "codes": set(), "agent": is_agent})
        if is_agent:
            old = self._agents.get(user["id"])
            if old:
                await self._sio.disconnect(old)  # newest agent wins
            self._agents[user["id"]] = sid
            await self._sio.emit("agent_status", {"online": True}, room=user_room(user["id"]))
        else:
            await self._sio.enter_room(sid, user_room(user["id"]))
            await self._sio.emit("agent_status", {"online": user["id"] in self._agents}, to=sid)

    async def disconnect(self, sid: str, *args) -> None:
        state = await self._sio.get_session(sid)
        user_id = state["user"]["id"]
        for code in list(state["codes"]):
            await self._leave(sid, state, code)
        if state.get("agent") and self._agents.get(user_id) == sid:
            del self._agents[user_id]
            self._agent_codes.pop(user_id, None)
            await self._sio.emit("agent_status", {"online": False}, room=user_room(user_id))

    async def ticket_create(self, sid: str, data: dict) -> dict:
        return await self._ticket_action(sid, data, create=True)

    async def ticket_dismiss(self, sid: str, data: dict) -> dict:
        return await self._ticket_action(sid, data, create=False)

    async def _ticket_action(self, sid: str, data: dict, create: bool) -> dict:
        state = await self._sio.get_session(sid)
        data = data or {}
        code = normalize_code(data.get("code"))
        if code not in state["codes"] or state.get("agent"):
            return {"ok": False, "error": "Join the session first"}
        if self._tickets is None:
            return {"ok": False, "error": "Jira tickets are turned off"}
        ticket_id = str(data.get("id") or "")
        try:
            if create:
                ticket = await self._tickets.create(
                    code, ticket_id, str(data.get("title") or ""), str(data.get("description") or "")
                )
            else:
                ticket = await self._tickets.dismiss(code, ticket_id)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        if ticket["status"] == "failed":
            return {"ok": False, "error": ticket.get("error"), "ticket": ticket}
        if ticket["status"] == "created" and self._store.get(code).summary is not None:
            # Created after the call was summarised: summarise again, so the record shows the ticket as done.
            self._schedule_analysis(code)
        return {"ok": True, "ticket": ticket}

    async def agent_ping(self, sid: str, data: dict) -> None:
        """The agent's heartbeat. After a reconnect it is a new connection: register it again and give it back
        the call it is still recording, or its audio (the other person) would be dropped."""
        state = await self._sio.get_session(sid)
        if not state.get("agent"):
            return
        user_id = state["user"]["id"]
        if self._agents.get(user_id) != sid:
            old = self._agents.get(user_id)
            if old is not None and self._sio.manager.is_connected(old, "/"):
                return  # another agent of this user is live; it keeps the role
            self._agents[user_id] = sid
            logger.info("Desktop agent registered again for user %s", user_id)
            await self._sio.emit("agent_status", {"online": True}, room=user_room(user_id))
        code = normalize_code((data or {}).get("code"))
        if not code or code in state["codes"]:
            return
        try:
            session = self._store.get(code)
        except SessionError:
            return
        if user_id in session.participants:
            state["codes"].add(code)
            self._agent_codes.setdefault(user_id, set()).add(code)
            logger.info("Desktop agent resumed recording session %s", code)

    async def agent_start(self, sid: str, data: dict) -> dict:
        state = await self._sio.get_session(sid)
        code = normalize_code((data or {}).get("code"))
        user_id = state["user"]["id"]
        if code not in state["codes"]:
            return {"ok": False, "error": "Join the session first"}
        agent_sid = self._agents.get(user_id)
        if agent_sid is None:
            return {"ok": False, "error": "Desktop agent is not running"}
        try:
            ack = await self._sio.call("start_recording", {"code": code}, to=agent_sid, timeout=AGENT_ACK_TIMEOUT_S)
        except Exception:
            return {"ok": False, "error": "Desktop agent did not respond"}
        if not (ack or {}).get("ok"):
            return {"ok": False, "error": (ack or {}).get("error") or "Desktop agent could not start"}
        agent_state = await self._sio.get_session(agent_sid)
        agent_state["codes"].add(code)
        self._agent_codes.setdefault(user_id, set()).add(code)
        return {"ok": True}

    async def agent_stop(self, sid: str, data: dict) -> dict:
        state = await self._sio.get_session(sid)
        await self._stop_agent(state["user"]["id"], normalize_code((data or {}).get("code")))
        return {"ok": True}

    async def _stop_agent(self, user_id: str, code: str) -> None:
        codes = self._agent_codes.get(user_id, set())
        agent_sid = self._agents.get(user_id)
        if code not in codes or agent_sid is None:
            return
        codes.discard(code)
        # The agent may still have chunks in flight; it keeps access until it sends stop_audio.
        await self._sio.emit("stop_recording", {"code": code}, to=agent_sid)

    async def join_session(self, sid: str, data: dict) -> dict:
        state = await self._sio.get_session(sid)
        try:
            session = self._store.join((data or {}).get("code"), state["user"], (data or {}).get("name"))
        except SessionError as exc:
            return {"ok": False, "error": str(exc)}
        state["codes"].add(session.code)
        if self._nba is not None:
            self._nba.warm()  # the caller may speak soon: have the hint prompt cached by then
        await self._sio.enter_room(sid, room_for(session.code))
        await self._broadcast_participants(session.code)
        return {"ok": True, "session": session.to_dict()}

    async def audio_chunk(self, sid: str, data: dict) -> None:
        state = await self._sio.get_session(sid)
        code = normalize_code((data or {}).get("code"))
        audio = (data or {}).get("audio")
        if code not in state["codes"] or not isinstance(audio, (bytes, bytearray)):
            return
        channels = (data or {}).get("channels", 1)
        if channels not in (1, 2) or not audio or len(audio) % (2 * channels):
            return
        if len(audio) > self._settings.max_chunk_bytes:
            return
        user = self._speaker(code, state["user"])
        key = (code, user["id"])
        if self._store.get(code).mode == "room":
            if key in self._muted:
                return
            mic = bytes(audio) if channels == 1 else split_stereo(bytes(audio))[0]  # the computer's audio is not used
            ts = self._chunk_ts(key, len(mic))
            self._room_audio.add(code, mic, ts)
            self._room_user[code] = user["id"]
            stream = self._streams.get(key) or self._new_stream(code, user, self._room_speakers(code, user["id"]))
            await stream.push(mic, ts)
            return
        if channels == 1:
            await self._stream_for(code, user).push(bytes(audio), self._chunk_ts(key, len(audio)))
            return
        mic, call = split_stereo(bytes(audio))
        ts = self._chunk_ts(key, len(mic))  # one clock for both channels: they were captured together
        if key in self._muted:  # my mic is muted: only the other person is transcribed
            await self._stream_for(code, self._call_speaker(code, user)).push(call, ts)
            return
        await asyncio.gather(
            self._stream_for(code, user).push(mic, ts),
            self._stream_for(code, self._call_speaker(code, user)).push(call, ts),
        )

    async def stop_audio(self, sid: str, data: dict) -> None:
        state = await self._sio.get_session(sid)
        code = normalize_code((data or {}).get("code"))
        if state.get("agent"):
            state["codes"].discard(code)
        await self._close_stream(code, state["user"]["id"])

    async def leave_session(self, sid: str, data: dict) -> None:
        state = await self._sio.get_session(sid)
        code = normalize_code((data or {}).get("code"))
        if code in state["codes"]:
            await self._leave(sid, state, code)

    def _speaker(self, code: str, user: dict) -> dict:
        """The user as recorded in this call: the name they entered when joining it."""
        participants = self._store.get(code).participants
        return {"id": user["id"], "name": participants.get(user["id"], user["name"])}

    def _call_speaker(self, code: str, user: dict) -> dict:
        """The user's call-audio channel: "Computer" unless renamed in this call."""
        speaker_id = call_speaker_id(user["id"])
        return {"id": speaker_id, "name": self._store.speaker_name(code, speaker_id, CALL_SPEAKER_NAME)}

    async def rename_speaker(self, sid: str, data: dict) -> dict:
        state = await self._sio.get_session(sid)
        code = normalize_code((data or {}).get("code"))
        speaker_id = str((data or {}).get("speaker_id") or "")
        if code not in state["codes"]:
            return {"ok": False, "error": "Join the session first"}
        try:
            self._store.rename_speaker(code, speaker_id, (data or {}).get("name"))
        except SessionError as exc:
            return {"ok": False, "error": str(exc)}
        name = self._store.speaker_name(code, speaker_id, "")
        live = self._stream_speakers.get((code, speaker_id))
        if live is not None:
            live["name"] = name  # lines still to come use the new name
        await self._sio.emit(
            "speaker_renamed", {"code": code, "speaker_id": speaker_id, "name": name}, room=room_for(code)
        )
        await self._broadcast_participants(code)
        return {"ok": True}

    def _room_speakers(self, code: str, user_id: str):
        """In-room mode: Deepgram's voice number -> the speaker it is recorded as (its dict is renamed in place)."""

        def speaker_for(label: int) -> dict:
            speaker_id = room_speaker_id(user_id, label + 1)
            speaker = self._stream_speakers.get((code, speaker_id))
            if speaker is None:
                name = self._store.speaker_name(code, speaker_id, room_speaker_name(speaker_id))
                speaker = self._stream_speakers[(code, speaker_id)] = {"id": speaker_id, "name": name}
            return speaker

        return speaker_for

    def _stream_for(self, code: str, speaker: dict) -> LiveStream:
        return self._streams.get((code, speaker["id"])) or self._new_stream(code, speaker)

    async def set_mode(self, sid: str, data: dict) -> dict:
        """Pick how the next recording tells speakers apart: call (by channel) or room (one mic, by voice)."""
        state = await self._sio.get_session(sid)
        code = normalize_code((data or {}).get("code"))
        if code not in state["codes"]:
            return {"ok": False, "error": "Join the session first"}
        if any(c == code for c, _ in self._streams):
            return {"ok": False, "error": "Stop the mic first"}
        try:
            self._store.set_mode(code, str((data or {}).get("mode") or ""))
        except SessionError as exc:
            return {"ok": False, "error": str(exc)}
        await self._sio.emit("mode_changed", {"code": code, "mode": self._store.get(code).mode}, room=room_for(code))
        return {"ok": True}

    async def mute_mic(self, sid: str, data: dict) -> dict:
        state = await self._sio.get_session(sid)
        code = normalize_code((data or {}).get("code"))
        if code not in state["codes"]:
            return {"ok": False, "error": "Join the session first"}
        key = (code, state["user"]["id"])
        if (data or {}).get("muted"):
            self._muted.add(key)
            # Save what was said so far; a new stream starts when the mic is unmuted.
            self._stream_speakers.pop(key, None)
            stream = self._streams.pop(key, None)
            if stream is not None:
                # Flushing the open sentence can take ~0.5 s (Deepgram); answer the mute at once.
                task = asyncio.create_task(stream.close())
                self._closing.add(task)
                task.add_done_callback(self._closing.discard)
        else:
            self._muted.discard(key)
        return {"ok": True}

    async def _leave(self, sid: str, state: dict, code: str) -> None:
        state["codes"].discard(code)
        if not state.get("agent"):
            self._muted.discard((code, state["user"]["id"]))
            # The web UI owns the call: when it leaves, the agent stops recording for it.
            await self._stop_agent(state["user"]["id"], code)
        await self._close_stream(code, state["user"]["id"])
        await self._sio.leave_room(sid, room_for(code))

    def _new_stream(self, code: str, user: dict, speaker_for=None) -> LiveStream:
        """A live stream of `user`'s audio. With `speaker_for` (in-room), it holds several voices."""
        masking = self._settings.pii_masking

        async def emit(payload: dict) -> None:
            if masking:
                payload = {**payload, "text": mask_text(payload["text"])}
            await self._sio.emit("transcript", payload, room=room_for(code))

        async def on_final(utterance: Utterance) -> None:
            if masking:
                utterance.text = mask_text(utterance.text)
            self._store.add_utterance(code, utterance)
            if self._nba is not None:
                try:
                    await self._nba.on_final(code, utterance)
                except Exception:  # a hint problem must never stop transcription
                    logger.exception("Live hints failed for session %s", code)
            if self._tickets is not None:
                try:
                    await self._tickets.on_final(code, utterance)
                except Exception:  # a ticket problem must never stop transcription
                    logger.exception("Ticket drafting failed for session %s", code)

        async def notify(payload: dict) -> None:
            await self._sio.emit("stt_status", payload, room=room_for(code))

        async def on_start(ts: float) -> None:
            # This speaker started a sentence: others in the call who are mid-sentence get a new bubble after it.
            for (c, speaker_id), other in list(self._streams.items()):
                if c == code and speaker_id != user["id"] and hasattr(other, "cut"):
                    await other.cut(ts)

        # In-room, a local engine with voice prints (sherpa) labels each line; another one puts all on Speaker 1.
        local_speaker = speaker_for(0) if speaker_for else user
        if isinstance(self._stt, DeepgramSTT):
            fallback = self._stt.fallback
            stream = DeepgramParticipantStream(
                code=code,
                speaker=local_speaker,
                stt=self._stt,
                sample_rate=self._settings.sample_rate,
                emit=emit,
                on_final=on_final,
                notify=notify,
                make_fallback=lambda: self._local_stream(code, local_speaker, fallback, emit, on_final, speaker_for),
                on_start=on_start,
                bubble_gap_s=self._settings.end_silence_ms / 1000,
                speaker_for=speaker_for,
            )
        else:
            stream = self._local_stream(code, local_speaker, self._stt, emit, on_final, speaker_for)
        self._streams[(code, user["id"])] = stream
        if speaker_for is None:
            self._stream_speakers[(code, user["id"])] = user
        return stream

    def _local_stream(self, code: str, user: dict, stt: STTProvider, emit, on_final, speaker_for=None) -> ParticipantStream:
        s = self._settings
        voices = stt.voices() if speaker_for is not None and hasattr(stt, "voices") else None
        segmenter = Segmenter(
            make_vad(s),
            sample_rate=s.sample_rate,
            # In-room, one line is one voice: a shorter pause ends it, so the next person gets their own line.
            end_silence_ms=min(s.end_silence_ms, ROOM_END_SILENCE_MS) if voices else s.end_silence_ms,
            max_utterance_ms=s.max_utterance_ms,
            min_speech_ms=s.min_speech_ms,
            preroll_ms=s.preroll_ms,
            partial_interval_ms=s.partial_interval_ms,
            partials_enabled=s.partials_enabled,
        )
        return ParticipantStream(
            code=code,
            speaker=user,
            segmenter=segmenter,
            stt=stt,
            sample_rate=s.sample_rate,
            language=s.stt_language,
            semaphore=self._semaphore,
            emit=emit,
            on_final=on_final,
            voices=voices,
            speaker_for=speaker_for if voices else None,
        )

    async def _close_stream(self, code: str, user_id: str) -> None:
        self._clocks.pop((code, user_id), None)
        mine = (user_id, call_speaker_id(user_id))
        for key in [k for k in self._stream_speakers if k[0] == code]:
            if key[1] in mine or key[1].startswith(f"{user_id}-room"):
                del self._stream_speakers[key]
        streams = [self._streams.pop((code, sid), None) for sid in (user_id, call_speaker_id(user_id))]
        await asyncio.gather(*(stream.close() for stream in streams if stream is not None))
        if any(s is not None for s in streams) and not any(c == code for c, _ in self._streams):
            self._schedule_analysis(code)

    def _schedule_analysis(self, code: str) -> None:
        """Summarise the call once nobody is streaming into it; a restarted mic postpones it. An in-room meeting's
        transcript is re-checked first."""
        if not self._settings.call_analysis and self._store.get(code).mode != "room":
            return
        previous = self._analysis.pop(code, None)
        if previous:
            previous.cancel()

        async def run() -> None:
            await asyncio.sleep(ANALYSIS_DELAY_S)
            if any(c == code for c, _ in self._streams):
                return  # recording again; analysed when it stops
            await self._recheck_room(code)
            if not self._settings.call_analysis:
                return
            await analyse_session(self._store, code, self._llm)
            if self._nba is not None:
                self._nba.warm()  # the summary replaced the cached hint prompt in a local model

        task = asyncio.create_task(run())
        self._analysis[code] = task
        task.add_done_callback(lambda t: self._analysis.pop(code, None) if self._analysis.get(code) is t else None)

    async def _recheck_room(self, code: str) -> None:
        session = self._store.get(code)
        if session.mode != "room":
            return
        user_id = self._room_user.pop(code, session.created_by)
        s = self._settings
        changed = await recheck(
            self._store, code, user_id, self._room_audio.take(code), self._stt, s.sample_rate,
            s.end_silence_ms / 1000, s.pii_masking,
        )
        if changed:
            transcript = [u.to_dict() for u in self._store.get(code).transcript]
            await self._sio.emit("transcript_reset", {"code": code, "transcript": transcript}, room=room_for(code))

    def _chunk_ts(self, key: tuple[str, str], nbytes: int) -> float:
        """Start time of this chunk: continue from the previous chunk unless the gap is too big."""
        duration = nbytes / 2 / self._settings.sample_rate
        arrival_start = time.time() - duration
        expected = self._clocks.get(key)
        ts = expected if expected is not None and abs(expected - arrival_start) <= CLOCK_RESYNC_S else arrival_start
        self._clocks[key] = ts + duration
        return ts

    async def _broadcast_participants(self, code: str) -> None:
        session = self._store.get(code)
        await self._sio.emit(
            "participants",
            {"code": code, "participants": session.to_dict()["participants"]},
            room=room_for(code),
        )
