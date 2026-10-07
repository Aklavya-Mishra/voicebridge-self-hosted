import pytest

from app.auth import AuthError, issue_token, verify_token
from app.config import Settings
from app.sessions import (
    CODE_ALPHABET,
    NotAParticipant,
    SessionFull,
    SessionNotFound,
    SessionStore,
    Utterance,
)


def test_token_round_trip(settings):
    token, user = issue_token(settings, "  Asha  ")
    assert user["name"] == "Asha"
    assert verify_token(settings, token) == user


@pytest.mark.parametrize("name", ["", "   ", "x" * 41])
def test_bad_names_rejected(settings, name):
    with pytest.raises(AuthError):
        issue_token(settings, name)


def test_token_from_other_secret_rejected(settings):
    token, _ = issue_token(Settings(jwt_secret="other"), "Asha")
    with pytest.raises(AuthError):
        verify_token(settings, token)


def test_missing_and_expired_token_rejected(settings):
    with pytest.raises(AuthError):
        verify_token(settings, None)
    token, _ = issue_token(Settings(jwt_secret="test-secret", jwt_ttl_seconds=-10), "Asha")
    with pytest.raises(AuthError):
        verify_token(settings, token)


A = {"id": "a", "name": "Asha"}
B = {"id": "b", "name": "Ben"}
C = {"id": "c", "name": "Cara"}


def test_create_and_join_two_participants():
    store = SessionStore()
    session = store.create(A)
    assert len(session.code) == 6 and set(session.code) <= set(CODE_ALPHABET)
    joined = store.join(session.code.lower(), B)  # codes are case-insensitive
    assert joined is session
    assert session.participants == {"a": "Asha", "b": "Ben"}
    assert store.join(session.code, A) is session  # rejoin is idempotent


def test_third_participant_rejected():
    store = SessionStore()
    session = store.create(A)
    store.join(session.code, B)
    with pytest.raises(SessionFull):
        store.join(session.code, C)


def test_unknown_code_and_non_participant():
    store = SessionStore()
    with pytest.raises(SessionNotFound):
        store.join("ZZZZZZ", A)
    session = store.create(A)
    with pytest.raises(NotAParticipant):
        store.require_participant(session.code, B)


def test_import_session_cannot_be_joined():
    store = SessionStore()
    session = store.create(A, kind="import")
    with pytest.raises(NotAParticipant):
        store.join(session.code, B)


def test_transcript_kept_in_start_order():
    store = SessionStore()
    session = store.create(A)
    for uid, start in (("u2", 5.0), ("u1", 1.0), ("u3", 3.0)):
        store.add_utterance(session.code, Utterance(uid, "a", "Asha", "hi", start, start + 1))
    assert [u.id for u in session.transcript] == ["u1", "u3", "u2"]
