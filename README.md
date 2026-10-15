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

Requirements: Python 3.11+, Node 20+, ffmpeg on PATH (for imports), and [Ollama](https://ollama.com) with
`ollama pull qwen2.5:7b` for call summaries (not needed if you set a Gemini or Anthropic key, see below).

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
2. **Start session** (or **Join** with a code) and enter **Your name in this call**: it is recorded on your
   lines. The other person is recorded under the name they enter if they use VoiceBridge, or as
   **Computer** if their voice comes from your computer's audio. Then **Start mic**:
   - desktop agent running: **Start** captures your mic and the computer's audio (the call);
   - no agent: **Mic only** transcribes just your side (the dialog says the agent is not running).
3. The chat shows your speech on the left and the other side on the right, one bubble per turn with
   its own time: a bubble keeps growing while the person talks and ends after 3 s of silence
   (`END_SILENCE_MS`), or as soon as the other side starts talking.
   The ✎ next to a name in the call header renames that speaker for the whole call (e.g. **Computer** to
   the caller's real name): earlier lines, new lines, the saved transcript and the summary.
4. **Transcript** downloads the call as text, one line per sentence:

   ```
   Aklavya: How can I help you?
   Computer: I was wondering if I can get help in flight booking.
   ```

   Each line carries the name its speaker was recorded with.
5. With the desktop agent, **Mute mic** stops transcribing only your side; the other person is still
   transcribed. **Unmute mic** brings your side back. The recording notice is shown once per call.
   **Leave** asks for confirmation; it also stops the agent's recording for that call.
6. Every call is saved as it happens (each finished sentence) in `backend/data/voicebridge.db` (SQLite;
   `DB_PATH` changes the location). **Past calls** on the home page lists them, newest first; opening one
   shows its summary and transcript (`Name: text`, one line per sentence) with a Back button.
7. Long numbers (card, phone, account) and emails are masked as `xxxxx` the moment a line is final, so they
   are never shown or saved in clear. When a call stops, it is analysed in the background (seconds on a
   cloud model, ~15-25 s for a short call on CPU): summary, purpose, outcome, action items and the caller's
   sentiment, plus masking of what patterns miss (expiry dates, CVVs, addresses,
   numbers read out in words). People's names are never masked. The sentiment is rated in a second short
   request (~1-3 s) from the caller's lines only (the call audio; in a call between two VoiceBridge users or an
   import, the person who did not start it), so a polite advisor does not soften an upset caller; a call where
   the caller never spoke has none.

### In-room meetings (one mic)

For people in one room with one microphone (no headset), click **In-room** instead of **Start mic** on the
call screen. Only the mic is used (with the agent, the computer's audio is ignored). Deepgram tells the voices
apart (`diarize=true`): lines go to **Speaker 1**, **Speaker 2**…, each renamable with ✎, and **Swap sides**
flips who is on the left. Live labels can be unsure in the first minute (in a test with two male voices, the
second person's first lines went to the first). So the meeting audio is kept in memory while recording (never
written to disk) and, on Stop, Deepgram hears all of it again; the transcript is rebuilt from those labels
(85.9% -> 96.8% of words on the right person in that test). Then the minutes are written: attendees,
discussion, decisions, next steps, open issues, and Jira tickets created during the meeting. Hints and
sentiment are off in this mode; Jira tickets work. Needs Deepgram: with the local fallback everyone is Speaker 1.

### Call analysis: which model

Set in `backend/.env`. The first available provider is used; if it fails (quota, network), the next is
tried:

1. **Gemini**, when `GEMINI_API_KEY` is set (`GEMINI_MODEL`, default `gemini-2.5-flash-lite`).
2. **Anthropic**, when `ANTHROPIC_API_KEY` is set (`ANTHROPIC_MODEL`, default `claude-opus-5-5`).
3. **Local model**, always: `LLM_MODEL` (default `qwen2.5:7b`) through Ollama at `OLLAMA_URL`, on the CPU
   only (`LLM_CPU_ONLY=true`). Free, nothing leaves the PC. Chosen after testing small open models on CPU:
   `qwen3:4b` miscopied card numbers and `gemma3:4b` padded its answers; `qwen2.5:7b` was exact, ~11 s.

The backend log names the order at start (`Call analysis with: ollama`). `CALL_ANALYSIS=false` turns the
summaries off; `PII_MASKING=false` turns masking off.

### Live hints (next best action)

During a call, an **Assist** panel next to the chat suggests what to say or ask next, after each line from
the other person (the call audio, "Computer" until renamed), and fills a **Customer info** card with what
they said (plan, device, budget, issue...). Hints are newest first; the latest is highlighted, `rule` marks
an instant hint from a rule, and alerts (cancellation, price objection, escalation...) have an orange edge.
Hints and Customer info are saved with the call and shown again under its summary in **Past calls**.

- **Call type**: never picked by hand, and no keywords decide it: the model detects it, with the hint, from
  each line of the other person, and the panel shows it (`Call type: Telecom`). When they change topic (a
  telecom issue, then a sales contract) it follows; a short reply to your question ("Yes.", "Baner, Pune.")
  keeps it. **General** covers greetings, thanks and goodbyes ("Hi, how are you?" gave "I am good, thank
  you. How can I assist you today?"), and any line the model places in no other type. Close types can
  still be mislabelled (sales vs marketing).
- Only the other person's lines trigger hints; calls between two VoiceBridge users get none.
- Each call type is one JSON file in `backend/app/nba/knowledge/`: intents with guidance for the advisor,
  facts to collect, and rules (phrases said, facts present, missing or equal to a value). Edit them to tune
  the hints; the backend checks them at start.
- Model: local `qwen2.5:7b` through Ollama on the CPU (`NBA_MODEL`), free, nothing leaves the PC; cloud keys
  are never used for hints. Every call type's guidance is in one fixed prompt (~2,400 tokens): read fresh it
  takes ~20 s on CPU, cached by Ollama ~0.5 s. Ollama drops that cache when it unloads the model (after
  `OLLAMA_KEEP_ALIVE`, 60m here; Ollama's default is 5m) or runs another prompt (the call summary), so the
  backend has the model read it again in the background at start, when a call is joined and after each
  summary. Measured on this PC: about 4-9 s per hint; up to ~28 s only when the caller speaks within ~20 s
  of such a warm-up starting (`NBA_TIMEOUT_S`, 60 s, covers it). Rule hints are instant. `python tools/bench_nba.py
  <model>` (in `backend/`) times a model on two sample calls.
- When Ollama is not running (or the model is not pulled), the panel says so and rule hints keep working.
  `NBA_ENABLED=false` turns live hints off.

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
| `mock` | No model; shows "(mock transcript: 1.4s of speech)". Handy for UI work |

Whisper on CPU: a sentence takes ~0.7 s with `STT_LANGUAGE=en`, ~1.4 s with auto-detect. The defaults
(`WHISPER_NUM_WORKERS=2`, `WHISPER_CPU_THREADS=1`, `OMP_WAIT_POLICY=PASSIVE`) come from measurements:
multi-threaded OpenMP busy-waiting made 4 threads ~20x slower than 1 on the dev machine.

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

- One backend process; calls are saved to a local SQLite file, not shared between servers.
- Sign-in is by name only: the same sign-in name is the same user (and sees the same past calls); the name
  shown in a transcript is the one entered for that call. Fine for one
  person; real accounts are needed before people share a deployment.
- Two participants per session.
- The desktop agent is Windows only. It hears all computer audio, so other sounds (videos, notifications)
  are transcribed as the other side; it also keeps transcribing your mic while you are muted in the call
  app. A full native endpoint is planned if the POC is approved.
- Proper names are often misheard.
- The local summary model is good at short calls; very long calls (well over an hour) exceed its context
  (`LLM_NUM_CTX`) and are summarised from what fits.
- Imports must be stereo with one speaker per channel; mono files are rejected.
- Use a headset: on speakers your mic can also pick up the other person.
