"""Download the sherpa-onnx model files used by STT_PROVIDER=sherpa and VAD_PROVIDER=sherpa (~500 MB, once).

Usage (from backend/):  python scripts/download_models.py [--dir models]

Files come from the k2-fsa/sherpa-onnx GitHub releases. Anything already present is skipped.
"""
import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
# (release tag, asset, what it unpacks to, minimum size in bytes of that file or folder)
MODELS = [
    ("asr-models", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", 600_000_000),
    ("asr-models", "silero_vad.onnx", "silero_vad.onnx", 600_000),
    ("speaker-segmentation-models", "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2", "sherpa-onnx-pyannote-segmentation-3-0", 5_000_000),
    ("speaker-recongition-models", "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx", "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx", 25_000_000),
]


def size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def fetch(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done * 100 // total:3d}% of {total >> 20} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="models")
    root = Path(parser.parse_args().dir)
    root.mkdir(parents=True, exist_ok=True)
    for tag, asset, target, min_size in MODELS:
        path = root / target
        if path.exists() and size_of(path) >= min_size:
            print(f"ok   {target}")
            continue
        print(f"get  {target}")
        download = root / asset
        fetch(f"{BASE}/{tag}/{asset}", download)
        if asset.endswith(".tar.bz2"):
            with tarfile.open(download) as tar:
                tar.extractall(root, filter="data")
            download.unlink()
        if size_of(path) < min_size:
            print(f"FAIL {target}: {size_of(path)} bytes, expected at least {min_size}", file=sys.stderr)
            return 1
        print(f"ok   {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
