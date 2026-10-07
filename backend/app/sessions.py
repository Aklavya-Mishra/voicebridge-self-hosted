"""In-memory call sessions: invite codes, participants and the ordered transcript.

Single-process only. A Redis-backed store with the same interface is the path to scale out.
"""
import bisect
import secrets
import time
import uuid
from dataclasses import asdict, dataclass, field

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

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "code": self.code,
            "kind": self.kind,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "participants": [{"id": uid, "name": name} for uid, name in self.participants.items()],
            "transcript": [u.to_dict() for u in self.transcript],
        }


def normalize_code(code: str | None) -> str:
    return (code or "").strip().upper()


class SessionStore:
    def __init__(self, max_participants: int = 2):
        self._max_participants = max_participants
        self._sessions: dict[str, Session] = {}

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
        return session

    def get(self, code: str | None) -> Session:
        session = self._sessions.get(normalize_code(code))
        if session is None:
            raise SessionNotFound("Session not found")
        return session

    def join(self, code: str | None, user: dict) -> Session:
        session = self.get(code)
        if user["id"] in session.participants:
            return session
        if session.kind != "live":
            raise NotAParticipant("Imported sessions cannot be joined")
        if len(session.participants) >= self._max_participants:
            raise SessionFull("Session already has the maximum number of participants")
        session.participants[user["id"]] = user["name"]
        return session

    def require_participant(self, code: str | None, user: dict) -> Session:
        session = self.get(code)
        if user["id"] not in session.participants:
            raise NotAParticipant("You are not a participant of this session")
        return session

    def add_guest(self, code: str, guest_id: str, name: str) -> None:
        self.get(code).participants[guest_id] = name

    def add_utterance(self, code: str, utterance: Utterance) -> None:
        bisect.insort(self.get(code).transcript, utterance, key=lambda u: u.start_ts)
