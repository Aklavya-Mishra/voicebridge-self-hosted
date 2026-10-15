"""Run a whole call against a running server, as the desktop agent would, and check what comes out.

Usage (from backend/, server running):
  python tools/e2e_call.py [--server http://localhost:8100] [--mode call|room] [--create-ticket]

call: streams tools/samples/tts/call_stereo.wav (left = you, right = the computer's audio) in real time.
room: streams tools/samples/tts/meeting_mono.wav as one mic (in-room mode).
Then it stops the mic, waits for the summary, and prints: the transcript, WER against the script, how many lines
went to the right person, the Jira draft (created in Jira with --create-ticket), the summary, and timings.
Exit code 1 when a check fails.
"""
import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import socketio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_stt import wer  # noqa: E402

SR = 16000
CHUNK_S = 0.1
TTS = Path(__file__).resolve().parent / "samples" / "tts"


def load(mode: str) -> tuple[bytes, int]:
    name = "call_stereo.wav" if mode == "call" else "meeting_mono.wav"
    with wave.open(str(TTS / name)) as w:
        return w.readframes(w.getnframes()), w.getnchannels()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8100")
    parser.add_argument("--mode", choices=("call", "room"), default="call")
    parser.add_argument("--create-ticket", action="store_true")
    parser.add_argument("--summary-timeout", type=float, default=300)
    args = parser.parse_args()
    script = json.loads((TTS / "script.json").read_text())
    failures: list[str] = []

    async with httpx.AsyncClient(base_url=args.server, timeout=30) as http:
        health = (await http.get("/api/health")).json()
        print(f"server: {health}")
        token = (await http.post("/api/auth/login", json={"name": "David"})).json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        code = (await http.post("/api/sessions", json={"mode": args.mode}, headers=headers)).json()["session"]["code"]

        sio = socketio.AsyncClient()
        lines, drafts, latencies = [], {}, []
        resets = []

        def on_transcript(d):
            if d["final"] and d["text"]:
                lines.append(d)
                latencies.append(time.time() - d["end_ts"])

        sio.on("transcript", on_transcript)
        sio.on("ticket_draft", lambda d: drafts.__setitem__(d["id"], d))
        sio.on("transcript_reset", lambda d: resets.append(d))
        await sio.connect(args.server, auth={"token": token}, transports=["websocket"])
        ack = await sio.call("join_session", {"code": code, "name": "David"})
        assert ack["ok"], ack
        if args.mode == "room":
            assert (await sio.call("set_mode", {"code": code, "mode": "room"}))["ok"]

        pcm, channels = load(args.mode)
        size = int(CHUNK_S * SR) * 2 * channels
        print(f"session {code}: streaming {len(pcm) / 2 / channels / SR:.0f} s of {args.mode} audio in real time…")
        start = time.time()
        for n, i in enumerate(range(0, len(pcm), size)):
            await sio.emit("audio_chunk", {"code": code, "audio": pcm[i : i + size], "channels": channels})
            await asyncio.sleep(max(0.0, start + (n + 1) * CHUNK_S - time.time()))
        await sio.call("stop_audio", {"code": code})
        stopped = time.time()

        summary = None
        while time.time() - stopped < args.summary_timeout:
            session = (await http.get(f"/api/sessions/{code}", headers=headers)).json()["session"]
            summary = session.get("summary")
            if summary and summary.get("status") != "pending":
                break
            await asyncio.sleep(2)
        summary_s = time.time() - stopped

        transcript = session["transcript"]
        print("\n--- transcript (saved)")
        for u in transcript:
            print(f"{u['speaker_name']}: {u['text']}")

        e, n = wer(" ".join(line["text"] for line in script), " ".join(u["text"] for u in transcript))
        print(f"\nWER {e / n:.1%} ({e} edits / {n} words)")
        if e / n > 0.10:
            failures.append(f"WER {e / n:.1%} > 10%")

        # Who said it: each saved line goes to the script line it overlaps most; speakers map to people by majority.
        votes: dict[tuple[str, str], int] = {}
        for u in transcript:
            mid = (u["start_ts"] + u["end_ts"]) / 2 - start
            ref = min(script, key=lambda line: 0 if line["start"] <= mid <= line["end"] else min(abs(mid - line["start"]), abs(mid - line["end"])))
            votes[(u["speaker_id"], ref["speaker"])] = votes.get((u["speaker_id"], ref["speaker"]), 0) + 1
        person = {}
        for (sid, who), _ in sorted(votes.items(), key=lambda kv: -kv[1]):
            person.setdefault(sid, who)
        right = sum(c for (sid, who), c in votes.items() if person[sid] == who)
        speakers = {u["speaker_id"] for u in transcript}
        print(f"speakers {len(speakers)} (2 real); lines on the right person {right}/{len(transcript)}")
        if len(speakers) != 2 or right < len(transcript) * 0.9:
            failures.append("speaker attribution")
        if args.mode == "room":
            print(f"in-room re-check: {'transcript rebuilt' if resets else 'no rebuild event'}")
        if latencies:
            print(f"final text after a bubble closes: median {np.median(latencies):.1f} s, max {max(latencies):.1f} s")

        print("\n--- Jira draft")
        if not drafts:
            print("none" + ("" if health.get("tickets") else " (Jira not configured)"))
            if health.get("tickets"):
                failures.append("no ticket drafted")
        for d in drafts.values():
            print(f"[{d['status']}] {d['title']}\n  {d['description']}")
            if args.create_ticket and d["status"] == "ready":
                res = await sio.call("ticket_create", {"code": code, "id": d["id"], "title": d["title"], "description": d["description"]}, timeout=60)
                ticket = res.get("ticket") or {}
                print(f"  created: {ticket.get('key')} {ticket.get('url')}" if res.get("ok") else f"  create failed: {res}")
                if not res.get("ok"):
                    failures.append("ticket create")

        print(f"\n--- summary ({summary_s:.0f} s after stop)")
        print(json.dumps(summary, indent=2) if summary else "none")
        if not summary or summary.get("status") != "ready":
            failures.append("no summary")
        await sio.disconnect()

    print("\nRESULT:", "PASS" if not failures else f"FAIL {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
