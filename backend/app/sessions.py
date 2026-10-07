"""Call sessions: invite codes, participants and the ordered transcript.

Kept in memory and, with a db_path, saved to SQLite as they change (every finished sentence), so calls
survive a backend restart. Single-process only; a shared store with the same interface is the path to
scale out.
"""
import bisect
import json
import re
import secrets
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I to avoid misreads
CODE_LENGTH = 6


class SessionError(Exception):
    status_code = 400


class SessionNotFound(SessionError):
    status_code = 404


class SessionFull(SessionError):
    status_code = 409


class NotAParticipant(SessionError):
    status_code = 403


class InvalidName(SessionError):
    status_code = 400


MAX_NAME_LENGTH = 40


@dataclass
class Utterance:
    id: str
    speaker_id: str
    speaker_name: str
    text: str
    start_ts: float
    end_ts: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Session:
    id: str
    code: str
    kind: str  # live | import
    created_by: str
    created_at: float
    participants: dict[str, str] = field(default_factory=dict)  # user id -> display name
    transcript: list[Utterance] = field(default_factory=list)
    summary: dict | None = None  # end-of-call analysis (app.analysis); status: pending | ready | unavailable
    # Names given to speakers who are not participants, e.g. the call audio ("<user id>-call") renamed from
    # "Computer" to the caller's real name.
    speaker_names: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "code": self.code,
            "kind": self.kind,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "participants": [{"id": uid, "name": name} for uid, name in self.participants.items()],
            "transcript": [u.to_dict() for u in self.transcript],
            "summary": self.summary,
            "speaker_names": self.speaker_names,
        }


def normalize_code(code: str | None) -> str:
    return (code or "").strip().upper()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    code TEXT PRIMARY KEY, id TEXT NOT NULL, kind TEXT NOT NULL, created_by TEXT NOT NULL,
    created_at REAL NOT NULL, participants TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS utterances (
    id TEXT PRIMARY KEY, code TEXT NOT NULL REFERENCES sessions(code), speaker_id TEXT NOT NULL,
    speaker_name TEXT NOT NULL, text TEXT NOT NULL, start_ts REAL NOT NULL, end_ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS utterances_by_code ON utterances(code, start_ts);
"""
# Columns added after the first release; created on databases that predate them.
_ADDED_COLUMNS = {"sessions": [("summary", "TEXT"), ("speaker_names", "TEXT")]}


class _Db:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Autocommit: each write is saved immediately. Writes come from the server's event loop thread.
        self._conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        for table, columns in _ADDED_COLUMNS.items():
            existing = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            for name, kind in columns:
                if name not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")

    def save_session(self, s: "Session") -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions"
            " (code, id, kind, created_by, created_at, participants, summary, speaker_names)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                s.code, s.id, s.kind, s.created_by, s.created_at, json.dumps(s.participants),
                json.dumps(s.summary) if s.summary is not None else None, json.dumps(s.speaker_names),
            ),
        )

    def save_utterance(self, code: str, u: Utterance) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO utterances VALUES (?, ?, ?, ?, ?, ?, ?)",
            (u.id, code, u.speaker_id, u.speaker_name, u.text, u.start_ts, u.end_ts),
        )

    def load(self) -> list["Session"]:
        rows = self._conn.execute(
            "SELECT code, id, kind, created_by, created_at, participants, summary, speaker_names FROM sessions"
        )
        sessions = {
            code: Session(
                id=sid, code=code, kind=kind, created_by=by, created_at=at, participants=json.loads(people),
                summary=json.loads(summary) if summary else None,
                speaker_names=json.loads(names) if names else {},
            )
            for code, sid, kind, by, at, people, summary, names in rows
        }
        rows = self._conn.execute(
            "SELECT id, code, speaker_id, speaker_name, text, start_ts, end_ts FROM utterances ORDER BY start_ts"
        )
        for uid, code, speaker_id, speaker_name, text, start, end in rows:
            if code in sessions:
                sessions[code].transcript.append(Utterance(uid, speaker_id, speaker_name, text, start, end))
        return list(sessions.values())


def _replace_name(value, old: str, new: str):
    """Swap a whole-word name inside the summary's texts."""
    if isinstance(value, str):
        return re.sub(rf"(?<!\w){re.escape(old)}(?!\w)", lambda _: new, value)
    if isinstance(value, list):
        return [_replace_name(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: _replace_name(v, old, new) for k, v in value.items()}
    return value


class SessionStore:
    def __init__(self, max_participants: int = 2, db_path: str | None = None):
        self._max_participants = max_participants
        self._sessions: dict[str, Session] = {}
        self._db = _Db(db_path) if db_path else None
        if self._db:
            self._sessions = {s.code: s for s in self._db.load()}

    def _save(self, session: Session) -> None:
        if self._db:
            self._db.save_session(session)

    def _new_code(self) -> str:
        while True:
            code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
            if code not in self._sessions:
                return code

    def create(self, user: dict, kind: str = "live") -> Session:
        session = Session(
            id=uuid.uuid4().hex,
            code=self._new_code(),
            kind=kind,
            created_by=user["id"],
            created_at=time.time(),
            participants={user["id"]: user["name"]},
        )
        self._sessions[session.code] = session
        self._save(session)
        return session

    def get(self, code: str | None) -> Session:
        session = self._sessions.get(normalize_code(code))
        if session is None:
            raise SessionNotFound("Session not found")
        return session

    def join(self, code: str | None, user: dict, name: str | None = None) -> Session:
        """Add the user to the session. `name` is their name in this call (recorded on their lines);
        without one, the name they already have in the session (or their sign-in name) is kept."""
        name = " ".join((name or "").split())
        if len(name) > MAX_NAME_LENGTH:
            raise InvalidName(f"Name must be at most {MAX_NAME_LENGTH} characters")
        session = self.get(code)
        if user["id"] in session.participants:
            if name and session.participants[user["id"]] != name:
                session.participants[user["id"]] = name
                self._save(session)
            return session
        if session.kind != "live":
            raise NotAParticipant("Imported sessions cannot be joined")
        if len(session.participants) >= self._max_participants:
            raise SessionFull("Session already has the maximum number of participants")
        session.participants[user["id"]] = name or user["name"]
        self._save(session)
        return session

    def require_participant(self, code: str | None, user: dict) -> Session:
        session = self.get(code)
        if user["id"] not in session.participants:
            raise NotAParticipant("You are not a participant of this session")
        return session

    def add_guest(self, code: str, guest_id: str, name: str) -> None:
        session = self.get(code)
        session.participants[guest_id] = name
        self._save(session)

    def add_utterance(self, code: str, utterance: Utterance) -> None:
        session = self.get(code)
        bisect.insort(session.transcript, utterance, key=lambda u: u.start_ts)
        if self._db:
            self._db.save_utterance(session.code, utterance)

    def update_utterance_text(self, code: str, utterance_id: str, text: str) -> None:
        """Replace a saved line's text (used to mask sensitive values after the call)."""
        session = self.get(code)
        for utterance in session.transcript:
            if utterance.id == utterance_id:
                utterance.text = text
                if self._db:
                    self._db.save_utterance(session.code, utterance)
                return

    def speaker_name(self, code: str, speaker_id: str, default: str) -> str:
        """The name a speaker is recorded with in this call."""
        session = self.get(code)
        return session.participants.get(speaker_id) or session.speaker_names.get(speaker_id) or default

    def rename_speaker(self, code: str, speaker_id: str, name: str) -> str:
        """Rename a speaker for the whole call: their saved lines, the summary and lines still to come.
        Returns the previous name."""
        name = " ".join((name or "").split())
        if not name or len(name) > MAX_NAME_LENGTH:
            raise InvalidName(f"Name must be 1-{MAX_NAME_LENGTH} characters")
        session = self.get(code)
        spoken = {u.speaker_id: u.speaker_name for u in session.transcript}
        if speaker_id in session.participants:
            old = session.participants[speaker_id]
            session.participants[speaker_id] = name
        elif speaker_id in session.speaker_names or speaker_id in spoken:
            old = session.speaker_names.get(speaker_id) or spoken[speaker_id]
            session.speaker_names[speaker_id] = name
        else:
            raise SessionNotFound("No such speaker in this call")

        for utterance in session.transcript:
            if utterance.speaker_id == speaker_id and utterance.speaker_name != name:
                utterance.speaker_name = name
                if self._db:
                    self._db.save_utterance(session.code, utterance)
        if session.summary and old != name:
            session.summary = _replace_name(session.summary, old, name)
        self._save(session)
        return old

    def set_summary(self, code: str, summary: dict | None) -> None:
        session = self.get(code)
        session.summary = summary
        self._save(session)

    def list_for(self, user_id: str) -> list[dict]:
        """The user's calls, newest first, with a one-line preview."""
        mine = sorted(
            (s for s in self._sessions.values() if user_id in s.participants),
            key=lambda s: s.created_at,
            reverse=True,
        )
        return [
            {
                "code": s.code,
                "kind": s.kind,
                "created_at": s.created_at,
                "participants": [{"id": uid, "name": name} for uid, name in s.participants.items()],
                "lines": len(s.transcript),
                "preview": s.transcript[0].text if s.transcript else "",
                "summary": {"status": s.summary.get("status"), "sentiment": s.summary.get("sentiment")}
                if s.summary
                else None,
            }
            for s in mine
        ]
