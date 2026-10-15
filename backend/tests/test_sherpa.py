"""Local engine (sherpa-onnx): pure logic always; the real models only when they are downloaded."""
import asyncio
import json
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.pipeline import ParticipantStream
from app.segmenter import Segmenter
from app.stt import make_stt
from app.stt.sherpa import VoiceTracker, words_from_tokens
from app.vad import EnergyVAD, make_vad
from tests.conftest import SR, RecordingSTT, chunks, silence, tone

BACKEND = Path(__file__).resolve().parents[1]
MODELS = BACKEND / "models"
TTS = BACKEND / "tools" / "samples" / "tts"
HAVE_MODELS = (MODELS / Settings.sherpa_asr_model).exists() and (MODELS / Settings.sherpa_embedding_model).exists()
needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="run python scripts/download_models.py")


def test_tokens_become_words_with_times():
    tokens = [" A", "fter", " ear", "ly", ",", " night"]
    words = words_from_tokens(tokens, [0.3, 0.5, 0.7, 0.9, 1.0, 1.2], end=1.6)
    assert words == [(0.3, 0.7, "After"), (0.7, 1.2, "early,"), (1.2, 1.6, "night")]


def test_sentencepiece_marker_starts_a_word():
    assert [w for _, _, w in words_from_tokens(["▁hi", "▁there"], [0.0, 0.4], end=1.0)] == ["hi", "there"]


def fake_embed(samples: np.ndarray) -> np.ndarray:
    """A voice print that is the line's pitch: tone(freq=220) and tone(freq=440) are two people."""
    spectrum = np.abs(np.fft.rfft(samples[:SR]))
    vec = np.zeros(8)
    vec[min(7, int(np.argmax(spectrum)) // 100)] = 1.0
    return vec


def test_voice_tracker_labels_people_in_order_of_first_appearance():
    tracker = VoiceTracker(fake_embed, threshold=0.5, sample_rate=SR)
    low, high = tone(1.5, freq=220), tone(1.5, freq=440)
    assert [tracker.label(p) for p in (low, high, low, high, high)] == [0, 1, 0, 1, 1]


def test_voice_tracker_keeps_the_last_voice_for_a_too_short_line():
    tracker = VoiceTracker(fake_embed, threshold=0.5, sample_rate=SR)
    tracker.label(tone(1.5, freq=440))
    tracker.label(tone(1.5, freq=220))
    assert tracker.label(tone(0.3, freq=440)) == 1  # too short to tell: still the previous line's voice


async def test_in_room_local_stream_puts_each_line_on_its_voice():
    speakers = {0: {"id": "u-room1", "name": "Speaker 1"}, 1: {"id": "u-room2", "name": "Speaker 2"}}
    emitted, stored = [], []

    async def emit(payload):
        emitted.append(payload)

    async def on_final(u):
        stored.append(u)

    stream = ParticipantStream(
        code="ABCDEF",
        speaker=speakers[0],
        segmenter=Segmenter(EnergyVAD(), sample_rate=SR, end_silence_ms=700, partials_enabled=False),
        stt=RecordingSTT(text="words"),
        sample_rate=SR,
        language=None,
        semaphore=asyncio.Semaphore(2),
        emit=emit,
        on_final=on_final,
        voices=VoiceTracker(fake_embed, threshold=0.5, sample_rate=SR),
        speaker_for=speakers.__getitem__,
    )
    ts = 10.0
    for chunk in chunks(tone(1.5, freq=220) + silence(1.0) + tone(1.5, freq=440) + silence(1.0)):
        await stream.push(chunk, ts)
        ts += len(chunk) / 2 / SR
    await stream.close()
    assert [u.speaker_id for u in stored] == ["u-room1", "u-room2"]
    assert [p["speaker"]["id"] for p in emitted if p["final"]] == ["u-room1", "u-room2"]


def test_unknown_model_dir_says_how_to_download(tmp_path):
    settings = Settings(stt_provider="sherpa", sherpa_model_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="download_models.py"):
        make_stt(settings)


# --- with the real models (CPU)


@pytest.fixture(scope="module")
def sherpa():
    return make_stt(Settings(stt_provider="sherpa", sherpa_model_dir=str(MODELS)))


def read(path: Path) -> list[bytes]:
    with wave.open(str(path)) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    return [data[:, c].copy().tobytes() for c in range(data.shape[1])]


@needs_models
async def test_parakeet_transcribes_english_with_punctuation(sherpa):
    (pcm,) = read(BACKEND / "tests" / "data" / "0.wav")
    text = await sherpa.transcribe(pcm, SR)
    assert text == "After early nightfall the yellow lamps would light up, here and there, the squalid quarter of the brothels."


@needs_models
async def test_parakeet_hears_nothing_in_silence(sherpa):
    assert await sherpa.transcribe(silence(2.0), SR) == ""


@needs_models
async def test_rejects_other_sample_rates(sherpa):
    with pytest.raises(ValueError):
        await sherpa.transcribe(silence(1.0), 8000)


@needs_models
@pytest.mark.skipif(not (TTS / "meeting_mono.wav").exists(), reason="run python tools/make_test_audio.py")
async def test_diarize_finds_both_people_of_a_meeting(sherpa):
    (pcm,) = read(TTS / "meeting_mono.wav")
    script = json.loads((TTS / "script.json").read_text())
    words = await sherpa.diarize(pcm, SR)
    assert len({w["speaker"] for w in words}) == 2
    for line in script:  # every line's words are on one voice, the same voice for the same person
        voices = {w["speaker"] for w in words if line["start"] - 0.2 <= w["start"] <= line["end"]}
        assert len(voices) == 1, line
        line["voice"] = voices.pop()
    by_person = {}
    for line in script:
        by_person.setdefault(line["speaker"], set()).add(line["voice"])
    assert all(len(v) == 1 for v in by_person.values()) and by_person["advisor"] != by_person["customer"]


@needs_models
def test_sherpa_vad_hears_speech_not_silence():
    vad = make_vad(replace(Settings(vad_provider="sherpa"), sherpa_model_dir=str(MODELS)))
    (pcm,) = read(BACKEND / "tests" / "data" / "0.wav")
    frame = vad.frame_samples * 2
    speech = [vad.is_speech(pcm[i : i + frame]) for i in range(0, len(pcm) - frame, frame)]
    quiet = [vad.is_speech(silence(0.032)) for _ in range(40)]
    assert sum(speech) > len(speech) * 0.6
    assert not any(quiet[10:])
