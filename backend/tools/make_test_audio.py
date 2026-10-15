"""Make test recordings with known words from Windows' offline voices (David, Zira): no download, no cloud.

Usage (from backend/, Windows):  python tools/make_test_audio.py [--out tools/samples/tts]

Writes, for a support call between an advisor (David) and a customer (Zira) that asks for a Jira ticket:
  call_stereo.wav   left = advisor, right = customer (what the desktop agent sends)
  meeting_mono.wav  both voices on one channel (in-room mode)
  script.json       every line: speaker, text, start/end seconds
"""
import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

SR = 16000
GAP_S = 1.2  # pause between turns; a little shorter than the 3 s bubble end, like a real conversation
VOICES = {"advisor": "Microsoft David Desktop", "customer": "Microsoft Zira Desktop"}
SCRIPT = [
    ("advisor", "Thank you for calling Acme broadband support. My name is David. How can I help you today?"),
    ("customer", "Hi David. My internet has been dropping every evening for the last three days."),
    ("advisor", "I am sorry to hear that. Can you tell me which router model you are using?"),
    ("customer", "It is the Acme X two hundred router. The light turns red around eight in the evening."),
    ("advisor", "Thank you. I can see a fault reported in your area. Let me check your line."),
    ("customer", "That would be great. I work from home, so this is really affecting my job."),
    ("advisor", "I understand. Your line shows packet loss after seven in the evening."),
    ("advisor", "Let me create a Jira ticket for the evening packet loss on your line so our network team investigates."),
    ("customer", "Thank you. How long will it take to fix?"),
    ("advisor", "The network team usually responds within forty eight hours. I will also send you an email update."),
    ("customer", "Okay, that works for me. Please also check if I can get a credit for the downtime."),
    ("advisor", "Yes, I will add a request for a one week service credit. Is there anything else?"),
    ("customer", "No, that is all. Thanks for your help."),
    ("advisor", "You are welcome. Have a great evening."),
]


def speak(text: str, voice: str, path: Path) -> np.ndarray:
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.SelectVoice('{voice}');"
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, 'Sixteen', 'Mono');"
        f"$s.SetOutputToWaveFile('{path}', $f); $s.Speak($env:VB_TEXT); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, env={**__import__("os").environ, "VB_TEXT": text})
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def write(path: Path, channels: list[np.ndarray]) -> None:
    data = np.stack(channels, axis=1) if len(channels) > 1 else channels[0]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(len(channels))
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.astype(np.int16).tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tools/samples/tts")
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    tracks = {who: [] for who in VOICES}
    lines, t = [], 1.0
    with tempfile.TemporaryDirectory() as tmp:
        for i, (who, text) in enumerate(SCRIPT):
            audio = speak(text, VOICES[who], Path(tmp) / f"{i}.wav")
            start = int(t * SR)
            for name, track in tracks.items():
                track.append((start, audio if name == who else None))
            lines.append({"speaker": who, "text": text, "start": round(t, 2), "end": round(t + len(audio) / SR, 2)})
            t += len(audio) / SR + GAP_S
    total = int((t + 1.0) * SR)
    channels = {}
    for name, track in tracks.items():
        buf = np.zeros(total, dtype=np.int32)
        for start, audio in track:
            if audio is not None:
                buf[start : start + len(audio)] += audio
        channels[name] = buf
    write(out / "call_stereo.wav", [channels["advisor"], channels["customer"]])
    write(out / "meeting_mono.wav", [np.clip(channels["advisor"] + channels["customer"], -32768, 32767)])
    (out / "script.json").write_text(json.dumps(lines, indent=2))
    print(f"Wrote {out}: {len(lines)} lines, {t:.0f} s")


if __name__ == "__main__":
    main()
