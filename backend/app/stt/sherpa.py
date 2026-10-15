"""Local transcription and voice separation with sherpa-onnx: CPU only, English, no key, nothing leaves the PC.

- transcribe(): NVIDIA Parakeet TDT 0.6B v2 (int8), one clip at a time; the text comes punctuated and cased.
- diarize(): the in-room re-check of a whole meeting. pyannote segmentation finds who spoke when, ERes2Net voice prints
  group those stretches into people, then Parakeet transcribes each stretch. Same output as DeepgramSTT.diarize.
- voices(): a VoiceTracker for live in-room labels, one voice print per finished line.

Model files come from scripts/download_models.py into SHERPA_MODEL_DIR.
"""
import asyncio
import logging
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from app.config import Settings

logger = logging.getLogger("voicebridge")

# Parakeet hears up to ~24 min at once, but memory grows with the clip: a longer stretch of one voice is cut.
MAX_CLIP_S = 30.0
# Stretches of one voice closer than this are transcribed together (more context, fewer cut words).
JOIN_GAP_S = 0.5
MIN_CLIP_S = 0.2
# A voice print from less speech than this is unreliable; such a line keeps the previous line's voice.
MIN_VOICE_S = 1.0
MAX_VOICES = 8
# In-room re-check: voice-print distance under which pyannote's stretches are one person. 0.5-0.9 all found the two
# people of the tools/make_test_audio.py meeting (100% of words right); 1.1 merged them.
CLUSTER_THRESHOLD = 0.7


def _samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _model_path(settings: Settings, name: str) -> str:
    path = Path(settings.sherpa_model_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"sherpa-onnx model not found: {path}. Run: python scripts/download_models.py")
    return str(path)


def words_from_tokens(tokens: list[str], timestamps: list[float], end: float) -> list[tuple[float, float, str]]:
    """Parakeet's sub-word tokens -> (start, end, word). A token starting with a space starts a new word; a word
    ends where the next one starts (or at `end`)."""
    words: list[list] = []
    for token, ts in zip(tokens, timestamps):
        piece = token.replace("▁", " ")
        if not words or piece.startswith(" "):
            words.append([ts, piece.strip()])
        else:
            words[-1][1] += piece
    out = []
    for i, (start, text) in enumerate(words):
        if text:
            stop = words[i + 1][0] if i + 1 < len(words) else max(end, start)
            out.append((float(start), float(stop), text))
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class VoiceTracker:
    """Live in-room labels: each finished line's voice print joins the closest known voice, or starts a new one.
    One tracker per recording; labels are 0, 1, 2... in order of first appearance."""

    def __init__(self, embed, threshold: float, sample_rate: int):
        self._embed = embed  # samples -> voice print
        self._threshold = threshold
        self._rate = sample_rate
        self._voices: list[tuple[np.ndarray, int]] = []  # (sum of voice prints, how many)
        self._last = 0
        self._lock = threading.Lock()  # two lines can finish at once, each in its own thread

    def label(self, pcm: bytes) -> int:
        samples = _samples(pcm)
        if len(samples) < MIN_VOICE_S * self._rate:
            return self._last
        print_ = self._embed(samples)
        with self._lock:
            return self._join(print_)

    def _join(self, print_: np.ndarray) -> int:
        best, score = None, -1.0
        for i, (total, n) in enumerate(self._voices):
            s = cosine(print_, total / n)
            if s > score:
                best, score = i, s
        if best is not None and (score >= self._threshold or len(self._voices) >= MAX_VOICES):
            total, n = self._voices[best]
            self._voices[best] = (total + print_, n + 1)
        else:
            self._voices.append((print_, 1))
            best = len(self._voices) - 1
        self._last = best
        return best


_diarizer = None  # in the child process: (models, OfflineSpeakerDiarization)


def who_spoke_when(models: tuple[str, str, int], pcm: bytes) -> list[tuple[float, float, int]]:
    """Runs in a child process: pyannote segmentation + ERes2Net voice prints -> sorted (start, end, voice)."""
    global _diarizer
    import sherpa_onnx as s

    if _diarizer is None or _diarizer[0] != models:
        segmentation, embedding, threads = models
        config = s.OfflineSpeakerDiarizationConfig(
            segmentation=s.OfflineSpeakerSegmentationModelConfig(
                pyannote=s.OfflineSpeakerSegmentationPyannoteModelConfig(model=segmentation), num_threads=threads
            ),
            embedding=s.SpeakerEmbeddingExtractorConfig(model=embedding, num_threads=threads),
            clustering=s.FastClusteringConfig(num_clusters=-1, threshold=CLUSTER_THRESHOLD),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        if not config.validate():
            raise RuntimeError("Invalid sherpa-onnx diarization config")
        _diarizer = (models, s.OfflineSpeakerDiarization(config))
    result = _diarizer[1].process(_samples(pcm)).sort_by_start_time()
    return [(float(r.start), float(r.end), int(r.speaker)) for r in result]


class SherpaSTT:
    name = "sherpa (Parakeet, local)"

    def __init__(self, settings: Settings):
        import sherpa_onnx

        self._sherpa = sherpa_onnx
        self._settings = settings
        self._rate = settings.sample_rate
        asr = Path(_model_path(settings, settings.sherpa_asr_model))
        logger.info("Loading sherpa-onnx model %s (%d thread(s) each)", asr.name, settings.sherpa_num_threads)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(next(asr.glob("encoder*.onnx"))),
            decoder=str(next(asr.glob("decoder*.onnx"))),
            joiner=str(next(asr.glob("joiner*.onnx"))),
            tokens=str(asr / "tokens.txt"),
            num_threads=settings.sherpa_num_threads,
            model_type="nemo_transducer",
            decoding_method="greedy_search",
        )
        self._diarizer: ProcessPoolExecutor | None = None  # in-room re-check, see diarize()
        self._extractor = None
        # The first inference is slower than later ones; pay that cost at startup.
        self._decode(np.zeros(self._rate, dtype=np.float32))

    # --- transcription

    def _decode(self, samples: np.ndarray):
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self._rate, samples)
        self._recognizer.decode_stream(stream)
        return stream.result

    def _words(self, samples: np.ndarray, offset: float) -> list[tuple[float, float, str]]:
        result = self._decode(samples)
        end = len(samples) / self._rate
        words = words_from_tokens(list(result.tokens), list(result.timestamps), end)
        if not words and result.text.strip():  # no timings: spread the words over the clip
            parts = result.text.split()
            step = end / len(parts)
            words = [(i * step, (i + 1) * step, w) for i, w in enumerate(parts)]
        return [(offset + s, offset + e, w) for s, e, w in words]

    async def transcribe(self, pcm: bytes, sample_rate: int, language: str | None = None) -> str:
        if sample_rate != self._rate:
            raise ValueError(f"sherpa expects {self._rate} Hz audio, got {sample_rate}")
        return await asyncio.to_thread(lambda: self._decode(_samples(pcm)).text.strip())

    # --- in-room: who spoke when

    def _embedding_config(self):
        return self._sherpa.SpeakerEmbeddingExtractorConfig(
            model=_model_path(self._settings, self._settings.sherpa_embedding_model),
            num_threads=self._settings.sherpa_num_threads,
        )

    def _pool(self) -> ProcessPoolExecutor:
        if self._diarizer is None:
            self._diarizer = ProcessPoolExecutor(max_workers=1)
        return self._diarizer

    def _stretches(self, raw: list[tuple[float, float, int]]) -> list[tuple[float, float, int]]:
        """pyannote's (start, end, voice) -> non-overlapping stretches in seconds; neighbours of one voice joined,
        long ones cut."""
        out: list[list] = []
        for seg_start, seg_end, voice in raw:
            start = max(seg_start, out[-1][1]) if out else seg_start  # overlapping speech goes to the earlier one
            if seg_end - start < MIN_CLIP_S:
                continue
            if out and out[-1][2] == voice and start - out[-1][1] <= JOIN_GAP_S:
                out[-1][1] = seg_end
            else:
                out.append([start, seg_end, voice])
        pieces = []
        for start, end, voice in out:
            while end - start > MAX_CLIP_S:
                pieces.append((start, start + MAX_CLIP_S, voice))
                start += MAX_CLIP_S
            pieces.append((start, end, voice))
        return pieces

    def _words_of(self, pcm: bytes, raw: list[tuple[float, float, int]]) -> list[dict]:
        samples = _samples(pcm)
        words = []
        for start, end, voice in self._stretches(raw):
            clip = samples[int(start * self._rate) : int(end * self._rate)]
            for s, e, w in self._words(clip, start):
                words.append({"start": s, "end": e, "word": w, "speaker": voice})
        return words

    async def diarize(self, pcm: bytes, sample_rate: int) -> list[dict]:
        """Words of a whole recording with who said them: [{start, end, word, speaker}], seconds from its start."""
        if sample_rate != self._rate:
            raise ValueError(f"sherpa expects {self._rate} Hz audio, got {sample_rate}")
        s = self._settings
        models = (
            _model_path(s, s.sherpa_segmentation_model), _model_path(s, s.sherpa_embedding_model), s.sherpa_num_threads
        )
        # sherpa's diarizer keeps Python's GIL while it runs (~6 s of a 100 s meeting measured): in this process it would
        # freeze every live call and request, so it runs in a child process. Transcription releases the GIL.
        raw = await asyncio.get_running_loop().run_in_executor(self._pool(), who_spoke_when, models, pcm)
        return await asyncio.to_thread(self._words_of, pcm, raw)

    def _embed(self, samples: np.ndarray) -> np.ndarray:
        if self._extractor is None:
            self._extractor = self._sherpa.SpeakerEmbeddingExtractor(self._embedding_config())
        stream = self._extractor.create_stream()
        stream.accept_waveform(self._rate, samples)
        stream.input_finished()
        return np.array(self._extractor.compute(stream), dtype=np.float32)

    def voices(self) -> VoiceTracker:
        return VoiceTracker(self._embed, self._settings.sherpa_speaker_threshold, self._rate)


class SherpaVAD:
    """Silero VAD through sherpa-onnx (no torch). Stateful, so create one per audio stream."""

    frame_samples = 512

    def __init__(self, settings: Settings, threshold: float = 0.5):
        import sherpa_onnx

        config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=_model_path(settings, settings.sherpa_vad_model),
                threshold=threshold,
                min_silence_duration=0.1,  # the segmenter decides when a sentence ends
                min_speech_duration=0.1,
                window_size=self.frame_samples,
            ),
            sample_rate=settings.sample_rate,
            num_threads=1,
        )
        self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)

    def is_speech(self, frame: bytes) -> bool:
        self._vad.accept_waveform(_samples(frame))
        speech = self._vad.is_speech_detected()
        while not self._vad.empty():  # its own speech segments are not used; drop them so memory stays flat
            self._vad.pop()
        return speech
