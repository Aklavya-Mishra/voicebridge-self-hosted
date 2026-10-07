# VoiceBridge

Transcribes a voice call (Teams, Zoom, Meet, a browser portal) as a chat: **your speech on the left,
the other person on the right**. Speakers are told apart by audio channel, not by guessing voices:

- **Desktop agent** (Windows, one person runs VoiceBridge): a small background program captures your mic
  (left channel) and everything the computer plays (right channel). Works with any call app; nothing is
  shared. The other person needs nothing installed.
- **Two endpoints**: both people run VoiceBridge in one session, each sending their own mic.
- **Import**: stereo recordings with one person per channel.

Design: [docs/specs/2026-10-07-voicebridge-v1-design.md](docs/specs/2026-10-07-voicebridge-v1-design.md)

```
agent/     Windows desktop agent (Python): mic + speaker loopback -> stereo chunks; started from the web UI
frontend/  React + Vite UI. v1 endpoint = browser mic capture (16 kHz PCM16, 100 ms chunks)
backend/   FastAPI + Socket.IO: auth, invite-code sessions, VAD segmentation, STT, import
```

## Run locally

Requirements: Python 3.11+, Node 20+, ffmpeg on PATH (for imports).

```bash
# backend
cd backend
python -m venv .venv
.venv/Scripts/activate          # Windows; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env            # then paste DEEPGRAM_API_KEY; without it, local Whisper is used
python run.py                   # http://localhost:8000

# frontend (second terminal)
cd frontend
npm install
npm run dev                     # http://localhost:5173
```

### Desktop agent (Windows)

Install: on the web app's home page click **Download agent (.exe)** and run it once. It lives in the system
tray and starts with Windows (turn that off in Task Manager > Startup apps). There is nothing to configure:
while you are signed in, the web app on the same PC hands the agent your sign-in over `127.0.0.1:47800`
(only the VoiceBridge web origin is accepted).

In a session: **Start mic** -> **Start**. Your mic is on the left; whatever the computer plays (the call, in
any app) is on the right as "Other person". Loopback hears all computer audio, so mute other sound during
calls. The agent uses the Windows default microphone and speakers. Log: `%APPDATA%\VoiceBridgegent.log`.

Build the .exe (also copies it to `frontend/public/downloads/` for the download link):

```powershell
cd agent
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File build.ps1
```

For a web app on another address, start the agent with `--origin https://your-host` (or set
`VOICEBRIDGE_WEB_ORIGINS`).

### Try a call with two endpoints

1. Open http://localhost:5173, sign in, **Start session**, note the invite code.
2. Second participant: another browser profile or machine on the LAN (`npm run dev` listens on all interfaces), sign in with another name, **Join** with the code.
   Or simulate one: `python tools/simulate_participant.py --code <CODE>` (add `--wav file.wav` to stream real speech).
3. Each person clicks **Start mic**. Your speech appears on the left, theirs on the right.

Browsers allow the microphone only on `localhost` or HTTPS. A second machine must reach the UI over HTTPS (or use a tunnel) to use its mic.

### Transcription

Set `STT_PROVIDER` in `backend/.env`:

| Provider | Setup |
|---|---|
| `deepgram` | default. Cloud, Nova-3, English + Hindi (`DEEPGRAM_LANGUAGE=multi`). `DEEPGRAM_API_KEY=...` from console.deepgram.com. Words appear ~0.3 s after speech. If Deepgram is unreachable (bad key, no credit, network), that speaker switches to `STT_FALLBACK` (default `whisper`) and the call shows a notice; audio Deepgram had not finalised is replayed locally, so nothing is lost. The next **Start mic** tries Deepgram again |
| `whisper` | Local faster-whisper, no key. Local faster-whisper, no key. First start downloads the model (`WHISPER_MODEL`, default `base`). Use `small`+ on a GPU (`WHISPER_DEVICE=cuda`) for better accuracy |
| `sherpa` | lowest latency, English. Local streaming model (sherpa-onnx), no key: words appear ~0.4 s after they are spoken. Each finished sentence is re-transcribed by Whisper for accuracy (`FINAL_PASS=whisper`, or `none`). Download the model once, below |
| `gemini` | `GEMINI_API_KEY=...`, or `VERTEX_PROJECT=...` with Google credentials |
| `mock` | no model; shows "(mock transcript: 1.4s of speech)". Handy for UI work |

Whisper on CPU: a sentence takes ~0.7 s with `STT_LANGUAGE=en`, ~1.4 s with auto-detect. The defaults
(`WHISPER_NUM_WORKERS=2`, `WHISPER_CPU_THREADS=1`, `OMP_WAIT_POLICY=PASSIVE`) come from measurements:
multi-threaded OpenMP busy-waiting made 4 threads ~20x slower than 1 on the dev machine.

Sherpa model download (310 MB, into `backend/models/`):

```bash
cd backend/models
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
tar xjf sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
```

Measured on the dev CPU, words trail speech by 0.40 s (median) with this model. The alternative
`sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06` (57 MB, same URL pattern) outputs punctuation and
casing itself but trails by 0.88 s; select it with `SHERPA_MODEL_DIR`. The UI merges consecutive
sentences from one speaker into one bubble; a new bubble opens on the other side when the speaker changes.

Deepgram bills every second of audio sent while the mic is on, silence included, per participant.
Nova-3 streaming costs $0.0048–0.0077/min (`en`) or $0.0058–0.0092/min (`multi`), so a one-hour
two-person call costs about $0.58–0.92.

## Tests

```bash
cd backend && .venv/Scripts/python -m pytest
cd frontend && npm test && npm run build && npm run lint
cd agent && .venv/Scripts/python -m pytest
```

## v1 limits

- One backend process (in-memory sessions). Redis-backed store is next for scale-out.
- Two participants per session.
- Imports must be stereo with one speaker per channel; mono files are rejected.
- Use a headset: on speakers your mic can also pick up the other person.
- Native desktop endpoint (Rust/Tauri) that runs without a browser tab is a later phase.
