"""Join a live session as a second participant and stream audio, for testing without a second machine.

Usage:
  python tools/simulate_participant.py --code K7P2QX [--name Ben] [--wav speech.wav] [--server http://localhost:8000]

Without --wav it streams synthetic speech-like tone bursts (works with STT_PROVIDER=mock).
"""
import argparse
import asyncio
import io

import httpx
import numpy as np
import socketio
from pydub import AudioSegment

SR = 16000
CHUNK_S = 0.1


def synthetic_audio() -> bytes:
    def tone(seconds: float, freq: float) -> np.ndarray:
        t = np.arange(int(seconds * SR)) / SR
        return np.sin(2 * np.pi * freq * t) * 0.3 * 32767

    gap = np.zeros(int(1.2 * SR))
    parts = [gap, tone(1.5, 220), gap, tone(2.5, 260), gap, tone(1.0, 300), gap]
    return np.concatenate(parts).astype(np.int16).tobytes()


def wav_audio(path: str) -> bytes:
    audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(SR).set_sample_width(2)
    return audio.raw_data


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--name", default="Simulated caller")
    parser.add_argument("--wav")
    parser.add_argument("--server", default="http://localhost:8000")
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.server) as http:
        token = (await http.post("/api/auth/login", json={"name": args.name})).json()["token"]

    sio = socketio.AsyncClient()
    sio.on("transcript", lambda d: d["final"] and d["text"] and print(f"[{d['speaker']['name']}] {d['text']}"))
    await sio.connect(args.server, auth={"token": token}, transports=["websocket"])
    ack = await sio.call("join_session", {"code": args.code})
    if not ack["ok"]:
        raise SystemExit(f"join failed: {ack['error']}")
    print(f"Joined {args.code} as {args.name}; streaming audio…")

    pcm = wav_audio(args.wav) if args.wav else synthetic_audio()
    size = int(CHUNK_S * SR) * 2
    for i in range(0, len(pcm), size):
        await sio.emit("audio_chunk", {"code": args.code, "audio": pcm[i : i + size]})
        await asyncio.sleep(CHUNK_S)  # real-time pacing
    await sio.emit("stop_audio", {"code": args.code})
    await asyncio.sleep(3)
    await sio.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
