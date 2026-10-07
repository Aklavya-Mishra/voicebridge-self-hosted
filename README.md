# VoiceBridge

Transcribes a voice call (Teams, Zoom, Meet, WhatsApp, a browser portal) as a chat: **your speech on the
left, the other person on the right**. Speakers are told apart by audio channel, not by guessing voices:

- **Desktop agent** (Windows, one person runs VoiceBridge): a small background program captures your mic
  (your side) and everything the computer plays (the other side). Works with any call app; nothing is
  shared. The other person needs nothing installed.
- **Two endpoints**: both people run VoiceBridge in one session, each sending their own browser mic.
- **Import**: stereo recordings with one person per channel.

Design: [docs/specs/2026-10-07-voicebridge-v1-design.md](docs/specs/2026-10-07-voicebridge-v1-design.md)

```
agent/     Windows desktop agent (Python, built as a tray .exe): mic + loopback of every output -> stereo chunks
frontend/  React + Vite UI: sessions, mirrored chat, transcript download; browser mic when no agent runs
backend/   FastAPI + Socket.IO: auth, invite-code sessions, per-channel STT (Deepgram + local fallback), import
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

## Using it

1. Open http://localhost:5173 and sign in with your name (the top-right pill shows the STT engine).
2. **Start session**, then **Start mic**:
   - desktop agent running: **Start** captures your mic and the computer's audio (the call);
   - no agent: **Mic only** transcribes just your side (the dialog says the agent is not running).
3. The chat shows your speech on the left and the other side on the right. Consecutive sentences from one
   speaker share a bubble; a pause of 3 s or more, or a speaker change, starts a new one.
4. **Transcript** downloads the call as text, one line per bubble:

   ```
   Aklavya: How can I help you?
   Computer: I was wondering if I can get help in flight booking.
   ```

   Your lines carry your sign-in name; everything from the other side is `Computer`.
5. **Leave** asks for confirmation; it also stops the agent's recording for that call.

### Desktop agent (Windows)

Install: on the web app's home page click **Download agent (.exe)** and run it once. A notification says it
is running; it lives in the system tray and starts with Windows (turn that off in Task Manager > Startup
apps). Starting it a second time just shows "already running". There is nothing to configure: while you
are signed in, the web app on the same PC hands the agent your sign-in over `127.0.0.1:47800` every 10 s
(only the VoiceBridge web origin is accepted), and the home page shows **Agent connected on this PC**.

The agent records the Windows default microphone and the loopback of **every** output device (speakers,
HDMI monitor, headset), since a call app may play on any of them. Every 10 s while recording it logs the
mic and call-audio levels to `%APPDATA%\VoiceBridge\agent.log`; a call-audio level of 0.00 means the
computer played nothing. Windows may warn "Windows protected your PC" on first run (the .exe is not code
signed): **More info > Run anyway**.

Build the .exe (also copies it to `frontend/public/downloads/` for the download link):

```powershell
cd agent
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File build.ps1      # -> agent\dist\VoiceBridgeAgent.exe
```

Run from source instead (logs to the console): `.venv\Scripts\python voicebridge_agent.py`.
For a web app on another address, add `--origin https://your-host` (or set `VOICEBRIDGE_WEB_ORIGINS`).

### Try a call with two endpoints

1. Open http://localhost:5173, sign in, **Start session**, note the invite code.
2. Second participant: another browser profile or machine on the LAN (`npm run dev` listens on all interfaces), sign in with another name, **Join** with the code.
   Or simulate one: `python tools/simulate_participant.py --code <CODE>` (add `--wav file.wav` to stream real speech).
3. Each person clicks **Start mic** > **Mic only**. Your speech appears on the left, theirs on the right.

Browsers allow the microphone only on `localhost` or HTTPS. A second machine must reach the UI over HTTPS (or use a tunnel) to use its mic.

### Transcription

Set `STT_PROVIDER` in `backend/.env` (`.env.example` uses `deepgram` with `whisper` as fallback):

| Provider | Setup |
|---|---|
| `deepgram` | Cloud, Nova-3, English + Hindi in one sentence (`DEEPGRAM_LANGUAGE=multi`; `en` for English only). `DEEPGRAM_API_KEY=...` from console.deepgram.com. Words appear ~0.3 s after speech. Each channel (your mic, the computer audio) has its own stream. If Deepgram is unreachable (bad key, no credit, network), that channel switches to `STT_FALLBACK` (default `whisper`) and the call shows a notice; audio Deepgram had not finalised is replayed locally, so nothing is lost. The next **Start** tries Deepgram again. File imports use Deepgram's pre-recorded API with the same fallback |
| `whisper` | Local faster-whisper, no key. First start downloads the model (`WHISPER_MODEL`: `.env.example` uses `small`, which handles Hindi; `small.en` is more accurate for English only). Leave `STT_LANGUAGE` empty to auto-detect per sentence. `WHISPER_DEVICE=cuda` on an NVIDIA GPU |
| `sherpa` | Lowest latency, English only. Local streaming model (sherpa-onnx), no key: words appear ~0.4 s after they are spoken. Each finished sentence is re-transcribed by Whisper for accuracy (`FINAL_PASS=whisper`, or `none`). Download the model once, below |
| `gemini` | `GEMINI_API_KEY=...`, or `VERTEX_PROJECT=...` with Google credentials |
| `mock` | No model; shows "(mock transcript: 1.4s of speech)". Handy for UI work |

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
casing itself but trails by 0.88 s; select it with `SHERPA_MODEL_DIR`.

Deepgram bills every second of audio sent while recording, silence included, per channel. Nova-3
streaming costs $0.0048–0.0077/min (`en`) or $0.0058–0.0092/min (`multi`), so a one-hour call (your mic +
the computer audio) costs about $0.58–0.92.

## Tests

```bash
cd backend && .venv/Scripts/python -m pytest
cd frontend && npm test && npm run build && npm run lint
cd agent && .venv/Scripts/python -m pytest
```

## v1 limits

- One backend process with in-memory sessions: transcripts are lost when the backend restarts (download
  them with **Transcript**).
- Two participants per session.
- The desktop agent is Windows only. It hears all computer audio, so other sounds (videos, notifications)
  are transcribed as the other side; it also keeps transcribing your mic while you are muted in the call
  app. A full native endpoint is planned if the POC is approved.
- Proper names are often misheard.
- Imports must be stereo with one speaker per channel; mono files are rejected.
- Use a headset: on speakers your mic can also pick up the other person.
