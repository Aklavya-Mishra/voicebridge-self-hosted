"""Renaming a speaker in a call (e.g. "Computer" -> "Rahul"): lines, summary and new lines all follow."""
import asyncio

import httpx
import pytest

from app.sessions import InvalidName, SessionError, SessionStore, Utterance
from tests.conftest import chunks, silence, tone
from tests.test_socket_e2e import connect, login, server, stereo  # noqa: F401  (server is a fixture)

ME = {"id": "u1", "name": "Aklavya"}


def seeded(tmp_path) -> tuple[SessionStore, str]:
    store = SessionStore(db_path=str(tmp_path / "vb.db"))
    code = store.create(ME).code
    store.add_utterance(code, Utterance("a", "u1", "Aklavya", "Hello, who is this?", 1.0, 2.0))
    store.add_utterance(code, Utterance("b", "u1-call", "Computer", "Rahul here.", 3.0, 4.0))
    store.set_summary(code, {
        "status": "ready", "summary": "Aklavya and Computer talk.", "purpose": "Computer calls about a flight.",
        "outcome": "Done.", "action_items": ["Email Computer the ticket"], "sentiment": "positive",
    })
    return store, code


def test_renaming_the_call_side_updates_lines_and_summary(tmp_path):
    store, code = seeded(tmp_path)
    store.rename_speaker(code, "u1-call", " Rahul ")

    session = SessionStore(db_path=str(tmp_path / "vb.db")).get(code)  # survives a restart
    assert [u.speaker_name for u in session.transcript] == ["Aklavya", "Rahul"]
    assert session.speaker_names == {"u1-call": "Rahul"}
    assert session.summary["summary"] == "Aklavya and Rahul talk."
    assert session.summary["purpose"] == "Rahul calls about a flight."
    assert session.summary["action_items"] == ["Email Rahul the ticket"]
    assert session.to_dict()["speaker_names"] == {"u1-call": "Rahul"}


def test_renaming_a_participant_updates_their_name(tmp_path):
    store, code = seeded(tmp_path)
    store.rename_speaker(code, "u1", "Aklavya M")
    session = store.get(code)
    assert session.participants == {"u1": "Aklavya M"}
    assert session.transcript[0].speaker_name == "Aklavya M"
    assert session.summary["summary"] == "Aklavya M and Computer talk."


@pytest.mark.parametrize("name", ["", "   ", "x" * 41])
def test_invalid_names_are_rejected(tmp_path, name):
    store, code = seeded(tmp_path)
    with pytest.raises(InvalidName):
        store.rename_speaker(code, "u1-call", name)


def test_unknown_speaker_is_rejected(tmp_path):
    store, code = seeded(tmp_path)
    with pytest.raises(SessionError):
        store.rename_speaker(code, "someone-else", "Bob")


async def test_rename_during_a_call_applies_to_past_and_new_lines(server):
    token, user = await login(server, "Aklavya")
    async with httpx.AsyncClient(base_url=server) as c:
        code = (await c.post("/api/sessions", headers={"Authorization": f"Bearer {token}"})).json()["session"]["code"]
    client, events = await connect(server, token)
    renamed = []
    client.on("speaker_renamed", renamed.append)
    assert (await client.call("join_session", {"code": code}))["ok"]
    call_id = f"{user['id']}-call"

    async def call_side_speaks():
        for chunk in chunks(stereo(silence(2.5), tone(1.0) + silence(1.5)), chunk_s=0.05):
            await client.emit("audio_chunk", {"code": code, "audio": chunk, "channels": 2})

    def finals():
        return [d for kind, d in events if kind == "transcript" and d["final"] and d["text"]]

    async def wait_for(predicate):
        for _ in range(50):
            if predicate():
                return
            await asyncio.sleep(0.1)

    await call_side_speaks()
    await wait_for(lambda: len(finals()) == 1)
    assert finals()[0]["speaker"]["name"] == "Computer"

    ack = await client.call("rename_speaker", {"code": code, "speaker_id": call_id, "name": "Rahul"})
    assert ack == {"ok": True}
    await wait_for(lambda: renamed)
    assert renamed[0] == {"code": code, "speaker_id": call_id, "name": "Rahul"}

    await call_side_speaks()
    await client.emit("stop_audio", {"code": code})
    await wait_for(lambda: len(finals()) == 2)
    assert finals()[1]["speaker"]["name"] == "Rahul"  # the live stream picked up the new name

    async with httpx.AsyncClient(base_url=server) as c:
        saved = (await c.get(f"/api/sessions/{code}", headers={"Authorization": f"Bearer {token}"})).json()["session"]
    assert [u["speaker_name"] for u in saved["transcript"]] == ["Rahul", "Rahul"]

    bad = await client.call("rename_speaker", {"code": code, "speaker_id": call_id, "name": ""})
    assert bad["ok"] is False
    await client.disconnect()


async def test_only_participants_can_rename(server):
    token, user = await login(server, "Aklavya")
    other_token, _ = await login(server, "Ben")
    async with httpx.AsyncClient(base_url=server) as c:
        code = (await c.post("/api/sessions", headers={"Authorization": f"Bearer {token}"})).json()["session"]["code"]
    outsider, _ = await connect(server, other_token)
    ack = await outsider.call("rename_speaker", {"code": code, "speaker_id": f"{user['id']}-call", "name": "X"})
    assert ack["ok"] is False
    await outsider.disconnect()
