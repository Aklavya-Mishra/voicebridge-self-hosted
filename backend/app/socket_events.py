"""Socket.IO events for live sessions.

Client -> server: join_session {code, name?}, rename_speaker {code, speaker_id, name} -> ack {ok, error?},
  audio_chunk {code, audio, channels?}, stop_audio {code}, leave_session {code}
  audio is PCM16 at the configured sample rate. channels=2 means interleaved stereo from one endpoint:
  left = the user's mic, right = the call audio (the remote person, who does not run VoiceBridge).
  Each channel gets its own STT stream; the channel decides the speaker.

Desktop agent (connects with auth {token, client: "agent"} using the web UI's token, handed over locally;
one per user):
  web -> server: agent_start {code} -> ack {ok, error?}; agent_stop {code} -> ack {ok}
  server -> agent: start_recording {code} -> ack {ok, error?}; stop_recording {code}
  server -> web: agent_status {online}
  The agent streams stereo audio_chunk (mic + speaker loopback) only into sessions it was started for.
Server -> client: participants {code, participants}, transcript {...see pipeline payload},
                  stt_status {code, speaker, engine, reason} when a speaker falls back to local STT,
                  speaker_renamed {code, speaker_id, name}
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
from app.pipeline import DeepgramParticipantStream, ParticipantStream, StreamingParticipantStream
from app.segmenter import Segmenter
from app.sessions import SessionError, SessionStore, Utterance, normalize_code
from app.stt import STTProvider
from app.stt.deepgram import DeepgramSTT
from app.vad import make_vad

logger = logging.getLogger("voicebridge")

LiveStream = ParticipantStream | StreamingParticipantStream | DeepgramParticipantStream

# Speaker for the call-audio (right) channel of a stereo endpoint.
CALL_SPEAKER_NAME = "Computer"

# How long the agent has to confirm start_recording.
AGENT_ACK_TIMEOUT_S = 5.0

# After the last stream of a call closes, wait this long before analysing it (the mic may restart).
ANALYSIS_DELAY_S = 1.0

# Re-anchor a stream's clock to arrival time when the gap is bigger than this (seconds).
CLOCK_RESYNC_S = 1.0


def room_for(code: str) -> str:
    return f"session:{code}"


def user_room(user_id: str) -> str:
    return f"user:{user_id}"


def call_speaker_id(user: dict) -> str:
    return f"{user['id']}-call"


def split_stereo(pcm: bytes) -> tuple[bytes, bytes]:
    frames = np.frombuffer(pcm, dtype=np.int16).reshape(-1, 2)
    return frames[:, 0].tobytes(), frames[:, 1].tobytes()


class SocketHandlers:
    def __init__(
        self, sio: socketio.AsyncServer, store: SessionStore, settings: Settings, stt: STTProvider, llm: LlmChain
    ):
        self._sio = sio
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
        events = ("connect", "disconnect", "join_session", "audio_chunk", "stop_audio", "leave_session",
                  "agent_start", "agent_stop", "rename_speaker")
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
        if channels == 1:
            await self._stream_for(code, user).push(bytes(audio), self._chunk_ts(key, len(audio)))
            return
        mic, call = split_stereo(bytes(audio))
        ts = self._chunk_ts(key, len(mic))  # one clock for both channels: they were captured together
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
        speaker_id = call_speaker_id(user)
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

    def _stream_for(self, code: str, speaker: dict) -> LiveStream:
        return self._streams.get((code, speaker["id"])) or self._new_stream(code, speaker)

    async def _leave(self, sid: str, state: dict, code: str) -> None:
        state["codes"].discard(code)
        if not state.get("agent"):
            # The web UI owns the call: when it leaves, the agent stops recording for it.
            await self._stop_agent(state["user"]["id"], code)
        await self._close_stream(code, state["user"]["id"])
        await self._sio.leave_room(sid, room_for(code))

    def _new_stream(self, code: str, user: dict) -> LiveStream:
        masking = self._settings.pii_masking

        async def emit(payload: dict) -> None:
            if masking:
                payload = {**payload, "text": mask_text(payload["text"])}
            await self._sio.emit("transcript", payload, room=room_for(code))

        async def on_final(utterance: Utterance) -> None:
            if masking:
                utterance.text = mask_text(utterance.text)
            self._store.add_utterance(code, utterance)

        async def notify(payload: dict) -> None:
            await self._sio.emit("stt_status", payload, room=room_for(code))

        if isinstance(self._stt, DeepgramSTT):
            fallback = self._stt.fallback
            stream = DeepgramParticipantStream(
                code=code,
                speaker=user,
                stt=self._stt,
                sample_rate=self._settings.sample_rate,
                emit=emit,
                on_final=on_final,
                notify=notify,
                make_fallback=lambda: self._local_stream(code, user, fallback, emit, on_final),
            )
        else:
            stream = self._local_stream(code, user, self._stt, emit, on_final)
        self._streams[(code, user["id"])] = stream
        self._stream_speakers[(code, user["id"])] = user
        return stream

    def _local_stream(
        self, code: str, user: dict, stt: STTProvider, emit, on_final
    ) -> ParticipantStream | StreamingParticipantStream:
        s = self._settings
        if hasattr(stt, "open_stream"):
            # Streaming model: it segments sentences itself, so no VAD/segmenter.
            return StreamingParticipantStream(
                code=code,
                speaker=user,
                stt=stt,
                sample_rate=s.sample_rate,
                language=s.stt_language,
                semaphore=self._semaphore,
                emit=emit,
                on_final=on_final,
            )

        segmenter = Segmenter(
            make_vad(s),
            sample_rate=s.sample_rate,
            end_silence_ms=s.end_silence_ms,
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
        )

    async def _close_stream(self, code: str, user_id: str) -> None:
        self._clocks.pop((code, user_id), None)
        for speaker_id in (user_id, f"{user_id}-call"):
            self._stream_speakers.pop((code, speaker_id), None)
        streams = [self._streams.pop((code, sid), None) for sid in (user_id, f"{user_id}-call")]
        await asyncio.gather(*(stream.close() for stream in streams if stream is not None))
        if any(s is not None for s in streams) and not any(c == code for c, _ in self._streams):
            self._schedule_analysis(code)

    def _schedule_analysis(self, code: str) -> None:
        """Summarise the call once nobody is streaming into it; a restarted mic postpones it."""
        if not self._settings.call_analysis:
            return
        previous = self._analysis.pop(code, None)
        if previous:
            previous.cancel()

        async def run() -> None:
            await asyncio.sleep(ANALYSIS_DELAY_S)
            if any(c == code for c, _ in self._streams):
                return  # recording again; analysed when it stops
            await analyse_session(self._store, code, self._llm)

        task = asyncio.create_task(run())
        self._analysis[code] = task
        task.add_done_callback(lambda t: self._analysis.pop(code, None) if self._analysis.get(code) is t else None)

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
