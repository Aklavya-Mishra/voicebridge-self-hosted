"""Live hints over a real server: the caller's line brings hints and customer info; mine bring nothing."""
import asyncio
from dataclasses import replace

import httpx
import pytest
import socketio
import uvicorn

from app.main import create_app
from tests.conftest import RecordingSTT, chunks, silence, tone
from tests.test_nba_engine import FakeLLM, answer
from tests.test_socket_e2e import free_port, login, stereo

EVENTS = ("transcript", "nba_hint", "nba_facts", "nba_domain", "nba_status")


@pytest.fixture
async def nba_server(settings):
    llm = FakeLLM(answer(facts=[{"key": "current_plan", "value": "Prepaid"}]))
    port = free_port()
    app = create_app(replace(settings, nba_enabled=True), stt=RecordingSTT(text="I want to cancel my plan"), nba_llm=llm)
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(srv.serve())
    while not srv.started:
        await asyncio.sleep(0.05)
    yield f"http://127.0.0.1:{port}", llm
    srv.should_exit = True
    await task


async def start_call(base: str, domain: str):
    token, _ = await login(base, "Aklavya")
    h = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(base_url=base) as c:
        code = (await c.post("/api/sessions", headers=h, json={"domain": domain})).json()["session"]["code"]
    client = socketio.AsyncClient()
    events: list[tuple[str, dict]] = []
    for name in EVENTS:
        client.on(name, lambda d, name=name: events.append((name, d)))
    await client.connect(base, auth={"token": token}, transports=["websocket"])
    assert (await client.call("join_session", {"code": code}))["ok"]
    return client, code, h, events


async def wait_for(predicate, tries: int = 60) -> bool:
    for _ in range(tries):
        if predicate():
            return True
        await asyncio.sleep(0.1)
    return False


def hint_ids(events):
    return [d["id"] for name, d in events if name == "nba_hint"]


async def test_caller_line_brings_hints_and_customer_info(nba_server):
    base, llm = nba_server
    client, code, h, events = await start_call(base, "telecom")
    for chunk in chunks(stereo(silence(2.0), tone(1.0) + silence(1.0)), chunk_s=0.05):
        await client.emit("audio_chunk", {"code": code, "audio": chunk, "channels": 2})
    await client.emit("stop_audio", {"code": code})

    assert await wait_for(lambda: "tel-prepaid" in hint_ids(events))
    assert hint_ids(events)[0] == "tel-cancel"
    assert any(d["kind"] == "ai" for name, d in events if name == "nba_hint")
    assert ("nba_facts", {"code": code, "facts": {"current_plan": "Prepaid"}}) in events
    assert "customer (Computer): I want to cancel my plan" in llm.calls[-1][1]

    async with httpx.AsyncClient(base_url=base) as c:
        saved = (await c.get(f"/api/sessions/{code}", headers=h)).json()["session"]
    assert saved["domain"] == "telecom"
    assert {"tel-cancel", "tel-prepaid"} <= {x["id"] for x in saved["nba"]["hints"]}
    rejoined = await client.call("join_session", {"code": code})
    assert rejoined["session"]["nba"]["facts"] == {"current_plan": "Prepaid"}
    await client.disconnect()


async def test_my_own_lines_bring_no_hints(nba_server):
    base, llm = nba_server
    client, code, _, events = await start_call(base, "telecom")
    for chunk in chunks(tone(1.0) + silence(1.0)):
        await client.emit("audio_chunk", {"code": code, "audio": chunk})
    await client.emit("stop_audio", {"code": code})
    assert await wait_for(lambda: any(n == "transcript" and d["final"] and d["text"] for n, d in events))
    await asyncio.sleep(0.3)
    assert [n for n, _ in events if n.startswith("nba_")] == [] and llm.calls == []
    await client.disconnect()


async def test_set_domain_over_the_socket(nba_server):
    base, _ = nba_server
    client, code, _, events = await start_call(base, "auto")
    assert (await client.call("set_domain", {"code": code, "domain": "sales"})) == {"ok": True}
    assert await wait_for(lambda: ("nba_domain", {"code": code, "domain": "sales", "auto": False}) in events)
    assert await client.call("set_domain", {"code": code, "domain": "cooking"}) == {"ok": False, "error": "Unknown call type"}
    assert (await client.call("set_domain", {"code": "ZZZZZZ", "domain": "sales"}))["ok"] is False
    await client.disconnect()


async def test_a_failing_hint_service_does_not_stop_transcription(nba_server, monkeypatch):
    from app.nba.service import NbaService

    async def boom(self, code, utterance):
        raise RuntimeError("hint service bug")

    monkeypatch.setattr(NbaService, "on_final", boom)
    base, _ = nba_server
    client, code, h, events = await start_call(base, "telecom")
    for chunk in chunks(stereo(silence(2.0), tone(1.0) + silence(1.0)), chunk_s=0.05):
        await client.emit("audio_chunk", {"code": code, "audio": chunk, "channels": 2})
    await client.emit("stop_audio", {"code": code})
    assert await wait_for(lambda: any(n == "transcript" and d["final"] and d["text"] for n, d in events))
    async with httpx.AsyncClient(base_url=base) as c:
        saved = (await c.get(f"/api/sessions/{code}", headers=h)).json()["session"]
    assert [u["text"] for u in saved["transcript"]] == ["I want to cancel my plan"]
    await client.disconnect()
