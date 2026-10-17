"""In-room mode: one mic in a meeting room hears everyone; Deepgram tells the voices apart.

Live, each word carries Deepgram's voice number and bubbles end when the voice changes (app.pipeline). Live labels
are unsure in the first minute (seen with two male voices: the second person's first lines went to the first). So
the meeting audio is kept in memory while recording (never written to disk) and, once the mic stops, Deepgram hears
all of it again: its labels are then much better (85.9% -> 96.8% of words on the right person in a test), and the
transcript of that span is rebuilt from them. Voice numbers are matched to the live speakers by the words they
share, so names given during the meeting stay.
"""
import logging
import uuid
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from app.analysis import mask_text
from app.sessions import SessionStore, Utterance, room_speaker_id, room_speaker_name
from app.stt import clean_transcript

logger = logging.getLogger("voicebridge")

# A chunk that does not continue the previous one within this many seconds starts a new piece of audio.
GAP_S = 1.0


@dataclass
class Piece:
    """Continuous audio of one recording: its first sample was captured at `start_ts` (server clock)."""

    start_ts: float
    pcm: bytearray = field(default_factory=bytearray)


class RoomRecorder:
    """The audio of each in-room meeting since its last re-check, in memory."""

    def __init__(self, sample_rate: int):
        self._rate = sample_rate
        self._pieces: dict[str, list[Piece]] = {}

    def add(self, code: str, pcm: bytes, ts: float) -> None:
        pieces = self._pieces.setdefault(code, [])
        if not pieces or abs(self._end(pieces[-1]) - ts) > GAP_S:
            pieces.append(Piece(ts))
        pieces[-1].pcm += pcm

    def take(self, code: str) -> list[Piece]:
        """The meeting's audio so far; it is forgotten here."""
        return self._pieces.pop(code, [])

    def _end(self, piece: Piece) -> float:
        return piece.start_ts + len(piece.pcm) / 2 / self._rate


def _at(pieces: list[Piece], rate: int, t: float) -> float:
    """Server time of second `t` of the pieces played one after another."""
    offset = 0.0
    for piece in pieces:
        length = len(piece.pcm) / 2 / rate
        if t < offset + length or piece is pieces[-1]:
            return piece.start_ts + t - offset
        offset += length
    return t


QUIET = 200  # sample level (PCM16) under which audio counts as silence
MAX_PAUSE_S = 0.6  # longer silences are cut before the re-check
KEEP_S = 0.15  # silence kept on each side of what is heard


def drop_silence(pieces: list[Piece], rate: int) -> list[Piece]:
    """The pieces with long silences cut out, each heard part a piece of its own (times stay right through `_at`).
    Long runs of digital silence, as in call audio while only I speak, made Deepgram hear one voice for four."""
    out: list[Piece] = []
    pad, pause = int(KEEP_S * rate), int(MAX_PAUSE_S * rate)
    for piece in pieces:
        samples = np.frombuffer(bytes(piece.pcm), dtype=np.int16)
        loud = np.flatnonzero(np.abs(samples.astype(np.int32)) > QUIET)
        if not len(loud):
            continue
        # heard parts: runs of loud samples whose gaps are shorter than `pause`
        breaks = np.flatnonzero(np.diff(loud) > pause)
        starts = np.concatenate(([loud[0]], loud[breaks + 1]))
        ends = np.concatenate((loud[breaks], [loud[-1]]))
        for a, b in zip(starts, ends):
            a, b = max(0, int(a) - pad), min(len(samples), int(b) + pad + 1)
            out.append(Piece(piece.start_ts + a / rate, bytearray(samples[a:b].tobytes())))
    return out


def _word_times(pieces: list[Piece], rate: int, start: float, end: float) -> dict:
    """Server times of a word: its piece is the one its middle falls in. A word's start can come a little before the
    piece it belongs to (cut audio), and on its own would land at the end of the piece before."""
    offset = 0.0
    for piece in pieces:
        length = len(piece.pcm) / 2 / rate
        if (start + end) / 2 < offset + length or piece is pieces[-1]:
            a = min(max(start - offset, 0.0), length)
            b = min(max(end - offset, a), length)
            return {"ts": piece.start_ts + a, "end_ts": piece.start_ts + b}
        offset += length
    return {"ts": start, "end_ts": end}


def speakers_by_label(words: list[dict], live: list[Utterance]) -> dict[int, str]:
    """Deepgram's voice numbers -> live speaker ids, by how many of each voice's words fall in that speaker's live
    lines. Most shared words first, one speaker per voice; a voice no live line explains gets none."""
    votes: Counter = Counter()
    for w in words:
        line = next((u for u in live if u.start_ts - 0.3 <= w["ts"] <= u.end_ts + 0.3), None)
        if line is not None:
            votes[(w["speaker"], line.speaker_id)] += 1
    mapping: dict[int, str] = {}
    for (label, speaker_id), _ in votes.most_common():
        if label not in mapping and speaker_id not in mapping.values():
            mapping[label] = speaker_id
    return mapping


def build_lines(words: list[dict], speaker_of: dict[int, dict], gap_s: float, masking: bool) -> list[Utterance]:
    """One line per turn: consecutive words of one voice, also split at `gap_s` of silence (as live bubbles)."""
    lines: list[Utterance] = []
    run: list[dict] = []

    def close() -> None:
        if not run:
            return
        speaker = speaker_of[run[0]["speaker"]]
        text = clean_transcript(" ".join(w["word"] for w in run))
        if masking:
            text = mask_text(text)
        if text:
            lines.append(
                Utterance(uuid.uuid4().hex[:12], speaker["id"], speaker["name"], text, run[0]["ts"], run[-1]["end_ts"])
            )
        run.clear()

    for w in words:
        if run and (w["speaker"] != run[-1]["speaker"] or w["ts"] - run[-1]["end_ts"] >= gap_s):
            close()
        run.append(w)
    close()
    return lines


async def recheck(
    store: SessionStore, code: str, user_id: str, pieces: list[Piece], stt, sample_rate: int, gap_s: float, masking: bool,
    speaker_id=room_speaker_id, default_name=room_speaker_name, only=None,
) -> bool:
    """Rebuild the transcript of the recorded span from the labels over the whole audio. Returns whether the
    transcript changed; on any failure the live transcript stays. For the call audio of a call, `speaker_id` and
    `default_name` give its voices ("Computer", "Computer 2"...) and `only` picks its lines (my mic's stay)."""
    if not pieces or not hasattr(stt, "diarize"):
        return False
    pieces = drop_silence(pieces, sample_rate) or pieces  # all quiet: nothing to cut
    pcm = b"".join(bytes(p.pcm) for p in pieces)
    try:
        raw = await stt.diarize(pcm, sample_rate)
    except Exception as exc:
        logger.warning("Re-check of meeting %s failed, keeping the live transcript: %s", code, exc)
        return False
    if not raw:
        return False  # nothing heard: keep whatever was shown live
    words = [{**w, **_word_times(pieces, sample_rate, w["start"], w["end"])} for w in raw]
    start = pieces[0].start_ts
    end = pieces[-1].start_ts + len(pieces[-1].pcm) / 2 / sample_rate
    session = store.get(code)
    live = [u for u in session.transcript if start - 1 <= u.start_ts <= end + 1 and (only is None or only(u))]
    mapping = speakers_by_label(words, live)
    used = {u.speaker_id for u in session.transcript} | set(mapping.values())
    number = 1
    for label in sorted({w["speaker"] for w in words} - set(mapping)):
        while speaker_id(user_id, number) in used:
            number += 1
        mapping[label] = speaker_id(user_id, number)
        used.add(mapping[label])
    speaker_of = {
        label: {"id": sid, "name": store.speaker_name(code, sid, default_name(sid))} for label, sid in mapping.items()
    }
    lines = build_lines(words, speaker_of, gap_s, masking)
    store.replace_utterances(code, start - 1, end + 1, lines, only=only)
    logger.info("Re-checked who spoke in meeting %s: %d live lines -> %d", code, len(live), len(lines))
    return True
