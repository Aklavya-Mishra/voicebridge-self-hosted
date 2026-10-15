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
    store: SessionStore, code: str, user_id: str, pieces: list[Piece], stt, sample_rate: int, gap_s: float, masking: bool
) -> bool:
    """Rebuild the transcript of the recorded span from Deepgram's labels over the whole audio. Returns whether the
    transcript changed; on any failure the live transcript stays."""
    if not pieces or not hasattr(stt, "diarize"):
        return False
    pcm = b"".join(bytes(p.pcm) for p in pieces)
    try:
        raw = await stt.diarize(pcm, sample_rate)
    except Exception as exc:
        logger.warning("Re-check of meeting %s failed, keeping the live transcript: %s", code, exc)
        return False
    if not raw:
        return False  # nothing heard: keep whatever was shown live
    words = [
        {**w, "ts": _at(pieces, sample_rate, w["start"]), "end_ts": _at(pieces, sample_rate, w["end"])} for w in raw
    ]
    start = pieces[0].start_ts
    end = pieces[-1].start_ts + len(pieces[-1].pcm) / 2 / sample_rate
    session = store.get(code)
    live = [u for u in session.transcript if start - 1 <= u.start_ts <= end + 1]
    mapping = speakers_by_label(words, live)
    used = {u.speaker_id for u in session.transcript} | set(mapping.values())
    number = 1
    for label in sorted({w["speaker"] for w in words} - set(mapping)):
        while room_speaker_id(user_id, number) in used:
            number += 1
        mapping[label] = room_speaker_id(user_id, number)
        used.add(mapping[label])
    speaker_of = {
        label: {"id": sid, "name": store.speaker_name(code, sid, room_speaker_name(sid))} for label, sid in mapping.items()
    }
    lines = build_lines(words, speaker_of, gap_s, masking)
    store.replace_utterances(code, start - 1, end + 1, lines)
    logger.info("Re-checked who spoke in meeting %s: %d live lines -> %d", code, len(live), len(lines))
    return True
