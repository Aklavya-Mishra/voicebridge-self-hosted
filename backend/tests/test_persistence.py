"""Saved calls: sessions and transcripts survive a backend restart."""
import httpx
import pytest

from app.auth import issue_token
from app.config import Settings
from app.main import create_app
from app.sessions import SessionStore, Utterance
from tests.conftest import RecordingSTT

ME = {"id": "u1", "name": "Aklavya"}


def utt(uid: str, text: str, start: float, speaker=ME) -> Utterance:
    return Utterance(uid, speaker["id"], speaker["name"], text, start, start + 1)


def test_sessions_and_transcripts_survive_a_restart(tmp_path):
    db = tmp_path / "vb.db"
    store = SessionStore(db_path=str(db))
    session = store.create(ME)
    store.add_utterance(session.code, utt("a", "How can I help you?", 10.0))
    store.add_utterance(session.code, utt("b", "Flight booking please.", 12.0, {"id": "u1-call", "name": "Other person"}))
    store.join(session.code, {"id": "u2", "name": "Ben"}, name="Ben K")

    reopened = SessionStore(db_path=str(db)).get(session.code)
    assert reopened.created_by == "u1"
    assert reopened.participants == {"u1": "Aklavya", "u2": "Ben K"}
    assert [u.text for u in reopened.transcript] == ["How can I help you?", "Flight booking please."]
    assert reopened.transcript[1].speaker_name == "Other person"


def test_new_codes_do_not_collide_with_saved_sessions(tmp_path):
    db = str(tmp_path / "vb.db")
    first = SessionStore(db_path=db).create(ME)
    store = SessionStore(db_path=db)
    assert all(store.create(ME).code != first.code for _ in range(20))


def test_list_for_user_newest_first_with_preview(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "vb.db"))
    old = store.create(ME)
    store.add_utterance(old.code, utt("a", "First call.", 1.0))
    new = store.create(ME)
    store.create({"id": "someone-else", "name": "X"})
    calls = store.list_for("u1")
    assert [c["code"] for c in calls] == [new.code, old.code]
    assert calls[1]["lines"] == 1 and calls[1]["preview"] == "First call."
    assert calls[0]["lines"] == 0 and calls[0]["preview"] == ""
    assert calls[0]["summary"] is None
    store.set_summary(old.code, {"status": "ready", "sentiment": "positive", "summary": "long text"})
    assert store.list_for("u1")[1]["summary"] == {"status": "ready", "sentiment": "positive"}


def test_without_db_path_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = SessionStore()
    store.add_utterance(store.create(ME).code, utt("a", "hi", 1.0))
    assert list(tmp_path.iterdir()) == []


def test_same_name_signs_in_as_the_same_user():
    settings = Settings(jwt_secret="s")
    _, a = issue_token(settings, "Aklavya")
    _, b = issue_token(settings, "  aklavya ")
    _, c = issue_token(settings, "Ben")
    assert a["id"] == b["id"] != c["id"]
    assert b["name"] == "aklavya"  # the name is shown as typed


@pytest.fixture
async def client(tmp_path):
    settings = Settings(jwt_secret="test-secret", db_path=str(tmp_path / "vb.db"))
    app = create_app(settings, stt=RecordingSTT())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_past_calls_api_lists_my_calls(client):
    token = (await client.post("/api/auth/login", json={"name": "Aklavya"})).json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    code = (await client.post("/api/sessions", headers=h)).json()["session"]["code"]
    calls = (await client.get("/api/sessions", headers=h)).json()["sessions"]
    assert [c["code"] for c in calls] == [code]
    assert (await client.get("/api/sessions")).status_code == 401


def test_older_tokens_map_to_the_stable_id_for_their_name():
    import time

    import jwt

    from app.auth import user_id_for, verify_token

    settings = Settings(jwt_secret="s")
    old = jwt.encode({"sub": "random-old-id", "name": "Aklavya", "exp": int(time.time()) + 60}, "s", algorithm="HS256")
    assert verify_token(settings, old) == {"id": user_id_for("Aklavya"), "name": "Aklavya"}


async def test_me_returns_the_server_side_identity(client):
    import time

    import jwt

    from app.auth import user_id_for

    old = jwt.encode({"sub": "random-old-id", "name": "Aklavya", "exp": int(time.time()) + 60}, "test-secret", algorithm="HS256")
    r = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {old}"})
    assert r.json() == {"user": {"id": user_id_for("Aklavya"), "name": "Aklavya"}}
    assert (await client.get("/api/auth/me")).status_code == 401
