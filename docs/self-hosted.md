# VoiceBridge self-hosted: what changed and why (2026-10-15)

Goal: the same VoiceBridge (desktop agent, web app, hints, Jira tickets, in-room meetings, summaries) at the least
running cost, so it can be benchmarked against the paid setup (Deepgram + a cloud model). Taken from `jira-test`
(`a839d2d`); everything else works as before. English only for now.

## Engines

| Job | Paid setup | Self-hosted default | Switch |
|---|---|---|---|
| Live and import speech-to-text | Deepgram Nova-3 (cloud) | Parakeet TDT 0.6B v2 int8, sherpa-onnx, CPU | `STT_PROVIDER=deepgram` / `sherpa` |
| Voice activity (local engines) | – | Silero through sherpa-onnx (no torch) | `VAD_PROVIDER=sherpa` / `energy` |
| In-room: who is speaking, live | Deepgram `diarize=true` | ERes2Net voice print per line, nearest known voice | follows `STT_PROVIDER` |
| In-room: re-check on Stop | Deepgram pre-recorded + diarize | pyannote segmentation 3.0 + ERes2Net clustering, then Parakeet | follows `STT_PROVIDER` |
| Summary / minutes | Gemini 2.5 Flash-Lite if `GEMINI_API_KEY` | `qwen2.5:7b` through Ollama, CPU | set or clear `GEMINI_API_KEY` |
| Live hints, Jira drafts | `qwen2.5:7b` (local already) | same | – |

Model files (~500 MB) come from `backend/scripts/download_models.py` (sherpa-onnx GitHub releases).

## How it fits

- `app/stt/sherpa.py` implements the same interface as the other engines: `transcribe(pcm, rate)` for each finished
  sentence and its live text (the existing segmenter cuts sentences), and `diarize(pcm, rate)` with the output of
  `DeepgramSTT.diarize`, so `app/room.py` rebuilds in-room transcripts unchanged.
- Live in-room labels: `ParticipantStream` takes an optional `VoiceTracker`; each finished line gets the closest known
  voice (cosine similarity >= `SHERPA_SPEAKER_THRESHOLD`) or a new one. In-room lines end after 1 s of silence
  (`ROOM_END_SILENCE_MS`) so a line holds one voice.
- sherpa's diarizer holds Python's GIL (~6 s for a 100 s meeting). It runs in a child process; transcription
  releases the GIL and runs in threads.

## Decisions

- **Parakeet over Nemotron streaming / Moonshine.** Parakeet gives punctuated, cased text with digits and fits the
  existing sentence-by-sentence path with no pipeline change. At ~28x real time on CPU, re-transcribing a growing
  sentence for live text is cheap. A streaming model would show live words sooner; it can be added behind the same
  setting later.
- **ERes2Net over CAM++** for voice prints. On the test meeting, CAM++ scored two lines of the same person as low as
  0.25 and two different people as high as 0.91, and found 6 voices for 2. ERes2Net: same person >= 0.74,
  different people <= 0.23; 2 voices, 100% of words right.
- **One Ollama context size.** Hints used 4096 tokens and summaries 16384. Ollama reloads a model whenever the size
  changes, and Ollama 0.40.4 hung a 16384 request indefinitely while the 4096 runner was loaded: the summary never came.
  `NBA_NUM_CTX` now defaults to `LLM_NUM_CTX`. Hint latency did not change (5.7–11.4 s per line, as before).
- **Kept** SQLite, Socket.IO, the React app and the agent as they are (no Postgres, MinIO, job workers or Tkinter
  from the earlier rebuild plan); they add infrastructure without lowering the bill.

## Cost per hour of call

- Paid: Deepgram streaming bills both channels, silence included: about $0.58–0.92 for a one-hour call
  (`en` is the cheaper end). A Gemini summary adds a small per-call charge; see Google's current pricing.
- Self-hosted: no per-minute charge. It needs one PC or server CPU: Parakeet uses ~0.04 s of CPU per second of
  speech per channel (plus re-runs for live text), and `qwen2.5:7b` needs ~6 GB RAM and 20–35 s of CPU per summary.
  The constraint is concurrent calls per machine, which the planned benchmark should measure.

## Tests added

- `backend/tests/test_sherpa.py`: token-to-word timing, voice tracker, in-room live labels, config errors; with the
  models: exact Parakeet transcript, silence, sample-rate check, two-person diarization, Silero VAD.
- `backend/tests/test_nba_config.py`: hints follow the summary context size.
- `backend/tools/eval_stt.py`: WER, real-time factor, in-room speaker accuracy, for `sherpa`, `whisper`, `deepgram`.
- `backend/tools/e2e_call.py`: a full call or meeting against a running server: transcript, WER, speaker attribution,
  latency, Jira draft (and creation with `--create-ticket`), summary.
- `backend/tools/make_test_audio.py`: the test call and meeting from Windows' offline voices.
