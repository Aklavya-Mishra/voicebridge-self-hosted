# VoiceBridge v1 — Design

## Goal

Two people on any voice call (Teams, Zoom, Meet, a browser portal) both run the VoiceBridge
endpoint. Each endpoint records only its own user's microphone. Speech becomes text and both
users see the same conversation as a chat: **my speech on the left, the other person on the
right**, mirrored on each side. Pre-recorded stereo files can be imported into the same view.

## Decisions (agreed in discussion)

| Topic | Decision |
|---|---|
| Audience | General product (own accounts, not tied to a tenant platform) |
| Pairing | Invite code: one user starts a session, the other joins with the code |
| Speaker identity | By channel, never by diarization. Live: each endpoint is its own channel. Import: stereo only, user says which channel is "me" |
| Mono imports | Rejected with a clear message (also rejected: stereo files whose two channels are identical) |
| VAD | Energy (dBFS) VAD by default, same idea as the reference service; Silero VAD optional |
| STT | Pluggable: `mock` (no keys, for dev), `gemini`, `whisper` (local, faster-whisper) |
| v1 endpoint | Runs in the browser UI (mic via `getUserMedia`). Native Rust/Tauri endpoint is a later phase |
| Group calls | Out of scope for v1 (max 2 participants per session) |

## Architecture

```
Browser endpoint + UI (React)                  Backend (Python, FastAPI + python-socketio)
 mic -> AudioWorklet -> 16 kHz PCM16 ---------> audio_chunk  -> ParticipantStream
                                                               -> Segmenter (VAD, per speaker)
                                                               -> STT provider
 chat view  <--------------------------------- transcript {partial|final, speaker, start_ts}
 import page -- POST /api/import (stereo) ----> importer: split channels -> Segmenter -> STT -> merge
```

### Backend units

- `config.py` — all settings from environment variables.
- `auth.py` — issue and verify HS256 JWTs (`sub` = user id, `name`). v1 login is display-name only.
- `sessions.py` — in-memory session store: invite codes, participants (max 2), ordered transcript.
- `audio_utils.py` — PCM/WAV helpers, dBFS.
- `vad.py` — `EnergyVAD`, optional `SileroVAD`, both on 512-sample (32 ms) frames.
- `segmenter.py` — turns one speaker's PCM stream into utterances. Speech starts an utterance
  (with 200 ms pre-roll); 700 ms of silence or 15 s of length ends it ("stable sentence").
  Emits periodic `partial` segments while speaking.
- `pipeline.py` — `ParticipantStream`: feeds the segmenter, runs STT (bounded concurrency),
  drops stale partials, emits transcript events, stores finals.
- `stt/` — provider interface plus `mock`, `gemini`, `whisper`; transcript clean-up.
- `importer.py` — stereo file -> per-channel segments -> STT -> merged transcript.
- `api.py` — REST: login, create session, get session, import.
- `socket_events.py` — Socket.IO: authenticated connect, `join_session`, `audio_chunk`,
  `stop_audio`, `leave_session`; emits `participants` and `transcript`.

### Contracts

Audio in (`audio_chunk`): `{code, audio: <binary PCM16 LE mono 16 kHz>}`. Chunks of ~100 ms.
The server assigns timestamps from its own clock, continuing the previous chunk's end so jitter
does not reorder speech; it resyncs if the gap exceeds 1 s.

Transcript out (`transcript`):

```json
{"code": "K7P2QX", "utterance_id": "a1b2c3d4e5f6", "speaker": {"id": "...", "name": "Asha"},
 "text": "Can you hear me?", "start_ts": 1760000000.12, "end_ts": 1760000001.9, "final": true}
```

A `final` with empty text means "drop this utterance" (noise, or an earlier partial that turned
out to be nothing). Clients sort bubbles by `start_ts` and place `speaker.id == my id` on the left.

## Error handling

- Socket connect without a valid JWT is refused.
- Joining a full session (2 other users) or an unknown code returns `{ok: false, error}`.
- Audio for a session the socket has not joined is ignored.
- STT failures are logged and produce an empty final (bubble removed), never a crash.
- Import: non-stereo, identical channels, undecodable or oversized files return HTTP 4xx.

## Testing

pytest unit tests for auth, session store, segmenter (synthetic tone/silence), pipeline
(fake STT, partial/final ordering), importer (generated stereo WAVs, mono rejection),
REST API via httpx, and a Socket.IO round trip against a live uvicorn server.

## Known limits of v1

- Single backend process (in-memory store). Redis-backed store is the path to scale out.
- Echo: on speakers (no headset) the mic can pick up the other person. Browser
  `echoCancellation` is enabled; a headset is recommended.
- Recording consent: the UI shows a recording indicator and a consent notice before the mic starts.
