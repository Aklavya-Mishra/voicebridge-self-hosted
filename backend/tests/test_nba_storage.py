"""The detected call type and hints are saved with the call, survive a restart and follow speaker renames."""
import sqlite3
from dataclasses import replace

import httpx
import pytest

from app.main import create_app
from app.sessions import SessionStore, Utterance, is_call_speaker
from tests.conftest import RecordingSTT

ME = {"id": "u1", "name": "Aklavya"}
HINT = {"id": "h1", "text": "Ask Computer for the area.", "kind": "ai", "intent": None, "priority": "suggestion", "ts": 1.0}


def test_new_calls_start_with_no_call_type_and_nothing_saved():
    s = SessionStore().create(ME)
    assert s.nba == {"hints": [], "facts": {}, "detected": None, "fact_log": {}, "domain_since": 0}
    assert s.to_dict()["nba"] == s.nba and "domain" not in s.to_dict()


def test_call_side_speaker_is_recognised():
    assert is_call_speaker("u1-call") and not is_call_speaker("u1")


def test_domain_and_hints_survive_a_restart(tmp_path):
    db = str(tmp_path / "vb.db")
    store = SessionStore(db_path=db)
    s = store.create(ME)
    nba = {"hints": [HINT], "facts": {"issue": "no internet"}, "detected": "telecom", "domain_since": 2}
    store.set_nba(s.code, nba)
    assert SessionStore(db_path=db).get(s.code).nba == {**nba, "fact_log": {}}


def test_database_from_before_hints_loads(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE sessions (code TEXT PRIMARY KEY, id TEXT NOT NULL, kind TEXT NOT NULL, created_by TEXT NOT NULL,"
        " created_at REAL NOT NULL, participants TEXT NOT NULL, summary TEXT, speaker_names TEXT);"
        """INSERT INTO sessions VALUES ('ABC234', 'id1', 'live', 'u1', 1.0, '{"u1": "Aklavya"}', NULL, '{}');"""
    )
    conn.close()
    s = SessionStore(db_path=str(db)).get("ABC234")
    assert s.nba == {"hints": [], "facts": {}, "detected": None, "fact_log": {}, "domain_since": 0}


def test_renaming_the_caller_updates_saved_hints_in_place():
    store = SessionStore()
    s = store.create(ME)
    store.add_utterance(s.code, Utterance("a", "u1-call", "Computer", "Hello", 1.0, 2.0))
    hints, facts = [dict(HINT)], {"contact": "Computer"}
    store.set_nba(s.code, {"hints": hints, "facts": facts, "detected": None})
    store.rename_speaker(s.code, "u1-call", "Rahul")
    assert hints[0]["text"] == "Ask Rahul for the area." and facts == {"contact": "Rahul"}
    assert store.get(s.code).nba["hints"] is hints  # same objects: the live hint service sees the rename


@pytest.fixture
async def client(settings):
    app = create_app(settings, stt=RecordingSTT())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


def test_database_with_a_picked_call_type_still_loads(tmp_path):
    """Databases from when the call type was picked by hand have a `domain` column; it is ignored."""
    db = tmp_path / "picked.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE sessions (code TEXT PRIMARY KEY, id TEXT NOT NULL, kind TEXT NOT NULL, created_by TEXT NOT NULL,"
        " created_at REAL NOT NULL, participants TEXT NOT NULL, summary TEXT, speaker_names TEXT, domain TEXT, nba TEXT);"
        """INSERT INTO sessions VALUES ('ABC234', 'id1', 'live', 'u1', 1.0, '{"u1": "Aklavya"}', NULL, '{}', 'sales',"""
        """ '{"hints": [], "facts": {}, "detected": null}');"""
    )
    conn.close()
    store = SessionStore(db_path=str(db))
    store.set_nba("ABC234", {**store.get("ABC234").nba, "detected": "general"})
    assert SessionStore(db_path=str(db)).get("ABC234").nba["detected"] == "general"


async def test_create_a_call(client):
    token = (await client.post("/api/auth/login", json={"name": "Aklavya"})).json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    made = (await client.post("/api/sessions", headers=h)).json()["session"]
    assert made["nba"]["detected"] is None and "domain" not in made


async def test_health_says_whether_hints_are_on(settings):
    for enabled in (False, True):
        app = create_app(replace(settings, nba_enabled=enabled), stt=RecordingSTT())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            assert (await c.get("/api/health")).json()["nba"] is enabled


def test_fact_history_survives_a_restart_and_follows_renames(tmp_path):
    db = str(tmp_path / "vb.db")
    store = SessionStore(db_path=db)
    s = store.create(ME)
    store.add_utterance(s.code, Utterance("a", "u1-call", "Computer", "Hello", 1.0, 2.0))
    store.set_nba(s.code, {"hints": [], "facts": {}, "detected": None, "fact_log": {"contact": ["Computer"]}})
    store.rename_speaker(s.code, "u1-call", "Rahul")
    assert SessionStore(db_path=db).get(s.code).nba["fact_log"] == {"contact": ["Rahul"]}
    assert SessionStore().create(ME).nba["fact_log"] == {}
