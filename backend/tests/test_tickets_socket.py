"""Jira tickets over a real server: a spoken request becomes a draft card; Create files it in (fake) Jira."""
import asyncio
from dataclasses import replace

import httpx
import pytest
import socketio
import uvicorn

from app.main import create_app
from tests.conftest import RecordingSTT, chunks, silence, tone
from tests.test_nba_engine import FakeLLM
from tests.test_socket_e2e import free_port, login, stereo
from tests.test_tickets import FakeJira

REQUEST = "Please create a Jira ticket where transcription has issues"


@pytest.fixture
async def ticket_server(settings):
    jira = FakeJira()
    llm = FakeLLM({"title": "Transcription has issues on calls", "description": "Words go missing."})
    jira_settings = replace(
        settings, jira_url="https://example.atlassian.net", jira_email="me@example.com", jira_api_token="t",
        jira_project_key="SCRUM",
    )
    port = free_port()
    app = create_app(jira_settings, stt=RecordingSTT(text=REQUEST), nba_llm=llm, jira=jira)
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(srv.serve())
    while not srv.started:
        await asyncio.sleep(0.05)
    yield f"http://127.0.0.1:{port}", jira
    srv.should_exit = True
    await task


async def wait_for(predicate, tries: int = 60) -> bool:
    for _ in range(tries):
        if predicate():
            return True
        await asyncio.sleep(0.1)
    return False


async def test_spoken_request_becomes_a_ticket_after_create(ticket_server):
    base, jira = ticket_server
    token, _ = await login(base, "Aklavya")
    h = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(base_url=base) as c:
        assert (await c.get("/api/health")).json()["tickets"] is True
        code = (await c.post("/api/sessions", headers=h)).json()["session"]["code"]
    client = socketio.AsyncClient()
    drafts: list[dict] = []
    client.on("ticket_draft", drafts.append)
    await client.connect(base, auth={"token": token}, transports=["websocket"])
    assert (await client.call("join_session", {"code": code}))["ok"]

    # The other person (call audio, right channel) asks for the ticket.
    for chunk in chunks(stereo(silence(2.0), tone(1.0) + silence(1.0)), chunk_s=0.05):
        await client.emit("audio_chunk", {"code": code, "audio": chunk, "channels": 2})
    await client.emit("stop_audio", {"code": code})

    assert await wait_for(lambda: drafts and drafts[-1]["status"] == "ready")
    assert drafts[0]["status"] == "drafting" and drafts[0]["requested_by"] == "Computer"
    ready = drafts[-1]
    assert ready["title"] == "Transcription has issues on calls"
    assert jira.created == []  # nothing is filed without the advisor's click

    ack = await client.call(
        "ticket_create", {"code": code, "id": ready["id"], "title": "Transcription drops words", "description": "x"}
    )
    assert ack["ok"] and ack["ticket"]["key"] == "SCRUM-5"
    assert jira.created == [("Transcription drops words", "x")]
    assert await wait_for(lambda: drafts[-1]["status"] == "created")

    rejoined = await client.call("join_session", {"code": code})
    assert rejoined["session"]["tickets"][0]["key"] == "SCRUM-5"
    assert (await client.call("ticket_dismiss", {"code": code, "id": ready["id"]}))["ok"] is False
    await client.disconnect()


async def test_ticket_events_need_the_call_joined(ticket_server):
    base, _ = ticket_server
    token, _ = await login(base, "Aklavya")
    client = socketio.AsyncClient()
    await client.connect(base, auth={"token": token}, transports=["websocket"])
    ack = await client.call("ticket_create", {"code": "ABCDEF", "id": "x", "title": "T"})
    assert ack == {"ok": False, "error": "Join the session first"}
    await client.disconnect()


async def test_ticket_created_after_the_summary_summarises_again(settings):
    from app.analysis import ANALYSIS_SCHEMA
    from app.llm import LlmChain

    class SummaryLLM:
        name = "ollama"

        def __init__(self):
            self.summary_inputs: list[str] = []

        async def json(self, system, user, schema):
            if schema is ANALYSIS_SCHEMA:
                self.summary_inputs.append(user)
                return {"summary": "s", "purpose": "p", "outcome": "No clear outcome", "pii": [],
                        "action_items": ["Aklavya will raise a Jira ticket for the crash"]}
            return {"sentiment": "negative"}

    summary_llm = SummaryLLM()
    jira_settings = replace(
        settings, call_analysis=True, jira_url="https://example.atlassian.net", jira_email="me@example.com",
        jira_api_token="t", jira_project_key="SCRUM",
    )
    port = free_port()
    app = create_app(
        jira_settings, stt=RecordingSTT(text=REQUEST), llm=LlmChain([summary_llm]),
        nba_llm=FakeLLM({"title": "Transcription has issues", "description": "d"}), jira=FakeJira(),
    )
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(srv.serve())
    while not srv.started:
        await asyncio.sleep(0.05)
    base = f"http://127.0.0.1:{port}"
    try:
        token, _ = await login(base, "Aklavya")
        h = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(base_url=base) as c:
            code = (await c.post("/api/sessions", headers=h)).json()["session"]["code"]
        client = socketio.AsyncClient()
        drafts: list[dict] = []
        client.on("ticket_draft", drafts.append)
        await client.connect(base, auth={"token": token}, transports=["websocket"])
        assert (await client.call("join_session", {"code": code}))["ok"]
        for chunk in chunks(stereo(silence(2.0), tone(1.0) + silence(1.0)), chunk_s=0.05):
            await client.emit("audio_chunk", {"code": code, "audio": chunk, "channels": 2})
        await client.emit("stop_audio", {"code": code})

        async def summary():
            async with httpx.AsyncClient(base_url=base) as c:
                return (await c.get(f"/api/sessions/{code}", headers=h)).json()["session"]["summary"]

        # The call ends and is summarised before anyone clicks Create.
        assert await wait_for(lambda: drafts and drafts[-1]["status"] == "ready" and len(summary_llm.summary_inputs) == 1)
        for _ in range(50):
            if (await summary() or {}).get("status") == "ready":
                break
            await asyncio.sleep(0.1)
        assert "Done during the call" not in summary_llm.summary_inputs[0]

        ack = await client.call("ticket_create", {"code": code, "id": drafts[-1]["id"], "title": "T", "description": ""})
        assert ack["ok"]
        assert await wait_for(lambda: len(summary_llm.summary_inputs) == 2)
        assert '- Jira ticket SCRUM-5 created: "T"' in summary_llm.summary_inputs[1]
        for _ in range(50):
            s = await summary()
            if s.get("status") == "ready" and s["outcome"] != "No clear outcome":
                break
            await asyncio.sleep(0.1)
        assert s["outcome"] == "Jira ticket SCRUM-5 created" and s["action_items"] == []
        await client.disconnect()
    finally:
        srv.should_exit = True
        await task
