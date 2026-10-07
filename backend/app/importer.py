"""Pre-recorded import: a stereo file with one speaker per channel.

Speaker identity comes from the channel, as in the live flow; there is no diarization.
"""
import asyncio
import io

import numpy as np
from pydub import AudioSegment

from app.config import Settings
from app.segmenter import FINAL, Segmenter
from app.sessions import Utterance
from app.stt import STTProvider, clean_transcript
from app.vad import make_vad

OTHER_SPEAKER_ID = "other"
OTHER_SPEAKER_NAME = "Other participant"
# Mean absolute sample difference below this means both channels carry the same (mono) audio.
IDENTICAL_CHANNEL_THRESHOLD = 8.0


class ImportRejected(Exception):
    pass


def _decode(data: bytes, sample_rate: int) -> AudioSegment:
    try:
        audio = AudioSegment.from_file(io.BytesIO(data))
    except Exception as exc:
        raise ImportRejected("Could not decode the audio file") from exc
    if audio.channels != 2:
        raise ImportRejected(
            f"The file has {audio.channels} channel(s). Only stereo files with one speaker per channel are supported."
        )
    return audio.set_frame_rate(sample_rate).set_sample_width(2)


def _check_channels_differ(left: AudioSegment, right: AudioSegment) -> None:
    a = np.frombuffer(left.raw_data, dtype=np.int16).astype(np.int32)
    b = np.frombuffer(right.raw_data, dtype=np.int16).astype(np.int32)
    if a.size and float(np.mean(np.abs(a - b))) < IDENTICAL_CHANNEL_THRESHOLD:
        raise ImportRejected(
            "Both channels contain the same audio (a mono recording saved as stereo). "
            "Speakers cannot be separated by channel."
        )


async def transcribe_stereo(
    data: bytes,
    me_channel: str,
    me: dict,
    settings: Settings,
    stt: STTProvider,
    started_at: float = 0.0,
) -> list[Utterance]:
    if me_channel not in ("left", "right"):
        raise ImportRejected("me_channel must be 'left' or 'right'")
    audio = await asyncio.to_thread(_decode, data, settings.sample_rate)
    left, right = audio.split_to_mono()
    _check_channels_differ(left, right)

    other = {"id": OTHER_SPEAKER_ID, "name": OTHER_SPEAKER_NAME}
    speakers = {"left": me, "right": other} if me_channel == "left" else {"left": other, "right": me}

    jobs = []
    for channel_name, channel_audio in (("left", left), ("right", right)):
        segmenter = Segmenter(
            make_vad(settings),
            sample_rate=settings.sample_rate,
            end_silence_ms=settings.end_silence_ms,
            max_utterance_ms=settings.max_utterance_ms,
            min_speech_ms=settings.min_speech_ms,
            preroll_ms=settings.preroll_ms,
            partials_enabled=False,
        )
        segments = segmenter.feed(channel_audio.raw_data, started_at) + segmenter.flush()
        jobs.extend((speakers[channel_name], s) for s in segments if s.kind == FINAL)

    semaphore = asyncio.Semaphore(settings.stt_concurrency)

    async def run(speaker: dict, segment) -> Utterance | None:
        async with semaphore:
            text = clean_transcript(await stt.transcribe(segment.pcm, settings.sample_rate, settings.stt_language))
        if not text:
            return None
        return Utterance(
            id=segment.utterance_id,
            speaker_id=speaker["id"],
            speaker_name=speaker["name"],
            text=text,
            start_ts=segment.start_ts,
            end_ts=segment.end_ts,
        )

    results = await asyncio.gather(*(run(speaker, segment) for speaker, segment in jobs))
    return sorted((u for u in results if u is not None), key=lambda u: u.start_ts)
