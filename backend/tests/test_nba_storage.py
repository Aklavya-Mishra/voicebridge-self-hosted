"""Call type and hints are saved with the call, survive a restart and follow speaker renames."""
import sqlite3
from dataclasses import replace

import httpx
import pytest

from app.main import create_app
from app.sessions import SessionStore, Utterance, is_call_speaker
from tests.conftest import RecordingSTT

ME = {"id": "u1", "name": "Aklavya"}
HINT = {"id": "h1", "text": "Ask Computer for the area.", "kind": "ai", "intent": None, "priority": "suggestion", "ts": 1.0}


def test_new_calls_default_to_auto_with_nothing_saved():
    s = SessionStore().create(ME)
    assert s.domain == "auto" and s.nba == {"hints": [], "facts": {}, "detected": None}
    assert s.to_dict()["domain"] == "auto" and s.to_dict()["nba"] == s.nba


def test_call_side_speaker_is_recognised():
    assert is_call_speaker("u1-call") and not is_call_speaker("u1")


def test_domain_and_hints_survive_a_restart(tmp_path):
    db = str(tmp_path / "vb.db")
    store = SessionStore(db_path=db)
    s = store.create(ME, domain="telecom")
    nba = {"hints": [HINT], "facts": {"issue": "no internet"}, "detected": None}
    store.set_nba(s.code, nba)
    again = SessionStore(db_path=db).get(s.code)
    assert again.domain == "telecom" and again.nba == nba
    store.set_domain(s.code, "sales")
    assert SessionStore(db_path=db).get(s.code).domain == "sales"


def test_database_from_before_hints_loads_as_auto(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE sessions (code TEXT PRIMARY KEY, id TEXT NOT NULL, kind TEXT NOT NULL, created_by TEXT NOT NULL,"
        " created_at REAL NOT NULL, participants TEXT NOT NULL, summary TEXT, speaker_names TEXT);"
        """INSERT INTO sessions VALUES ('ABC234', 'id1', 'live', 'u1', 1.0, '{"u1": "Aklavya"}', NULL, '{}');"""
    )
    conn.close()
    s = SessionStore(db_path=str(db)).get("ABC234")
    assert s.domain == "auto" and s.nba == {"hints": [], "facts": {}, "detected": None}


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


async def test_create_a_call_with_a_call_type(client):
    token = (await client.post("/api/auth/login", json={"name": "Aklavya"})).json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    made = await client.post("/api/sessions", headers=h, json={"domain": "tech_support"})
    assert made.json()["session"]["domain"] == "tech_support"
    assert (await client.post("/api/sessions", headers=h)).json()["session"]["domain"] == "auto"
    assert (await client.post("/api/sessions", headers=h, json={"domain": "cooking"})).status_code == 422


async def test_health_says_whether_hints_are_on(settings):
    for enabled in (False, True):
        app = create_app(replace(settings, nba_enabled=enabled), stt=RecordingSTT())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            assert (await c.get("/api/health")).json()["nba"] is enabled
