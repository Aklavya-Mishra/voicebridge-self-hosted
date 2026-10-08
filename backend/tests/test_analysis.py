"""End-of-call analysis: summary + PII masking, and the LLM provider chain (Gemini > Anthropic > local)."""
import asyncio
import json

import httpx
import pytest

from app.analysis import ANALYSIS_SCHEMA, analyse_session, apply_pii, mask_text
from app.config import Settings
from app.llm import LlmChain, OllamaLLM, make_llm_chain
from app.sessions import SessionStore, Utterance

ME = {"id": "u1", "name": "Aklavya"}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Card 4532 7781 2290 1146 expiry", "Card xxxxx expiry"),
        ("call me on +91 98765 43210 please", "call me on xxxxx please"),
        ("account 004512339871", "account xxxxx"),
        ("email rahul.verma82@gmail.com.", "email xxxxx."),
        ("Card is four five three two, seven seven eight one, two two nine zero.", "Card is xxxxx."),
        ("my number is nine eight double seven six five four three two one", "my number is xxxxx"),
    ],
)
def test_mask_text_hides_long_numbers_and_emails(text, expected):
    assert mask_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "The fare is 6,450 rupees.", "Give me 10 minutes.", "PNR is K7X2QP.", "Flight at 8:30 in 2026.", "Call 100.",
        "One or two people, maybe three.",
    ],
)
def test_mask_text_keeps_ordinary_numbers(text):
    assert mask_text(text) == text


def test_apply_pii_masks_listed_values_but_never_names():
    text = "Rahul Verma here, expiry 09/28, CVV 123, address 12 MG Road."
    out = apply_pii(text, ["09/28", "rahul verma", "123", "12 MG Road", "x"], keep=["Aklavya", "Computer", "Rahul Verma"])
    assert out == "Rahul Verma here, expiry xxxxx, CVV xxxxx, address xxxxx."


class FakeLLM:
    def __init__(self, name, result=None, fail=False):
        self.name = name
        self.calls = 0
        self._result = result
        self._fail = fail

    async def json(self, system, user, schema):
        self.calls += 1
        if self._fail:
            raise RuntimeError(f"{self.name} down")
        return self._result


def test_chain_order_follows_available_keys():
    both = make_llm_chain(Settings(gemini_api_key="g", anthropic_api_key="a"))
    assert both.names == ["gemini", "anthropic", "ollama"]
    assert make_llm_chain(Settings(anthropic_api_key="a")).names == ["anthropic", "ollama"]
    assert make_llm_chain(Settings()).names == ["ollama"]


async def test_chain_falls_back_to_the_next_provider():
    first, second = FakeLLM("gemini", fail=True), FakeLLM("ollama", result={"ok": 1})
    result, used = await LlmChain([first, second]).json("s", "u", {})
    assert result == {"ok": 1} and used == "ollama" and first.calls == 1


async def test_ollama_runs_on_cpu_with_the_schema():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"summary": "hi"}'}})

    llm = OllamaLLM(Settings(), transport=httpx.MockTransport(handler))
    assert await llm.json("system", "user", {"type": "object"}) == {"summary": "hi"}
    assert seen["model"] == "qwen2.5:7b"
    assert seen["options"]["num_gpu"] == 0
    assert seen["think"] is False and seen["stream"] is False
    assert seen["format"] == {"type": "object"}
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]


ANALYSIS = {
    "summary": "Rahul booked a flight.",
    "purpose": "Book a flight",
    "action_items": ["Email the ticket"],
    "outcome": "Booked",
    "sentiment": "positive",
    "pii": ["09/28", "Rahul Verma"],
}


def seeded_store(tmp_path) -> tuple[SessionStore, str]:
    store = SessionStore(db_path=str(tmp_path / "vb.db"))
    code = store.create(ME).code
    lines = [("u1", "Aklavya", "How can I help?"), ("u1-call", "Computer", "Rahul Verma here, card expiry 09/28.")]
    for i, (sid, name, text) in enumerate(lines):
        store.add_utterance(code, Utterance(str(i), sid, name, text, float(i), i + 0.5))
    return store, code


async def test_analysis_saves_summary_and_masks_pii_except_names(tmp_path):
    store, code = seeded_store(tmp_path)
    llm = FakeLLM("ollama", result=ANALYSIS)
    await analyse_session(store, code, LlmChain([llm]))

    reopened = SessionStore(db_path=str(tmp_path / "vb.db")).get(code)
    assert reopened.summary["status"] == "ready"
    assert reopened.summary["summary"] == "Rahul booked a flight."
    assert reopened.summary["provider"] == "ollama"
    assert "pii" not in reopened.summary  # the values themselves are not kept
    assert reopened.transcript[1].text == "Rahul Verma here, card expiry xxxxx."


async def test_analysis_unavailable_when_every_provider_fails(tmp_path):
    store, code = seeded_store(tmp_path)
    await analyse_session(store, code, LlmChain([FakeLLM("ollama", fail=True)]))
    assert store.get(code).summary == {"status": "unavailable"}


async def test_empty_call_is_not_sent_to_the_model(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "vb.db"))
    code = store.create(ME).code
    llm = FakeLLM("ollama", result=ANALYSIS)
    await analyse_session(store, code, LlmChain([llm]))
    assert llm.calls == 0 and store.get(code).summary is None


def test_schema_asks_for_every_field():
    assert set(ANALYSIS_SCHEMA["required"]) == set(ANALYSIS) and ANALYSIS_SCHEMA["additionalProperties"] is False


async def test_call_end_triggers_analysis_and_live_text_is_masked(settings):
    """Over a real server: the final line is masked as it is shown and stored; stopping runs the analysis."""
    import socket

    import socketio
    import uvicorn

    from app.main import create_app
    from tests.conftest import RecordingSTT, chunks, silence, tone
    from tests.test_socket_e2e import login

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    llm = FakeLLM("ollama", result=dict(ANALYSIS, pii=[]))
    settings = Settings(jwt_secret="test-secret", call_analysis=True)
    app = create_app(settings, stt=RecordingSTT(text="my card is 4532 7781 2290 1146"), llm=LlmChain([llm]))
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
        finals = []
        client.on("transcript", lambda d: d["final"] and d["text"] and finals.append(d["text"]))
        await client.connect(base, auth={"token": token}, transports=["websocket"])
        await client.call("join_session", {"code": code})
        for chunk in chunks(tone(1.0) + silence(1.0)):
            await client.emit("audio_chunk", {"code": code, "audio": chunk})
        await client.emit("stop_audio", {"code": code})
        for _ in range(60):
            async with httpx.AsyncClient(base_url=base) as c:
                session = (await c.get(f"/api/sessions/{code}", headers=h)).json()["session"]
            if (session.get("summary") or {}).get("status") == "ready":
                break
            await asyncio.sleep(0.1)
        assert finals == ["my card is xxxxx"]
        assert session["transcript"][0]["text"] == "my card is xxxxx"
        assert session["summary"]["summary"] == "Rahul booked a flight."
        await client.disconnect()
    finally:
        srv.should_exit = True
        await task


async def test_anthropic_provider_sends_structured_output_request(monkeypatch):
    """Through the real SDK, against a local stand-in for the Anthropic API."""
    from aiohttp import web

    from app.llm import AnthropicLLM

    seen = {}

    async def messages(request):
        seen["body"] = await request.json()
        seen["beta"] = request.headers.get("anthropic-beta", "")
        return web.json_response({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": '{"summary": "ok"}'}],
            "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    app = web.Application()
    app.router.add_post("/v1/messages", messages)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{port}")
    try:
        llm = AnthropicLLM(Settings(anthropic_api_key="test-key"))
        assert await llm.json("sys", "transcript", {"type": "object"}) == {"summary": "ok"}
    finally:
        await runner.cleanup()
    body = seen["body"]
    assert body["model"] == "claude-opus-5-5" and body["system"] == "sys"
    assert body["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert body["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in seen["beta"]


async def test_gemini_provider_sends_json_schema_request(monkeypatch):
    """Through the real google-genai SDK, against a local stand-in for the Gemini API."""
    from aiohttp import web

    from app.llm import GeminiLLM

    seen = {}

    async def generate(request):
        seen["path"] = request.path
        seen["body"] = await request.json()
        return web.json_response({"candidates": [{"content": {"role": "model", "parts": [{"text": '{"summary": "ok"}'}]}}]})

    app = web.Application()
    app.router.add_post("/{tail:.*}", generate)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", f"http://127.0.0.1:{port}")
    try:
        llm = GeminiLLM(Settings(gemini_api_key="test-key"))
        assert await llm.json("sys", "transcript", {"type": "object"}) == {"summary": "ok"}
    finally:
        await runner.cleanup()
    assert seen["path"].endswith("models/gemini-2.5-flash-lite:generateContent")
    config = seen["body"]["generationConfig"]
    assert config["responseMimeType"] == "application/json" and config["responseJsonSchema"] == {"type": "object"}
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "sys"


async def test_summary_text_is_masked_too(tmp_path):
    store, code = seeded_store(tmp_path)
    leaky = dict(
        ANALYSIS,
        summary="Rahul Verma paid with card 1234 5678 9012 3456, expiry 09/28.",
        action_items=["Email the ticket to rahul@x.com"],
    )
    await analyse_session(store, code, LlmChain([FakeLLM("ollama", result=leaky)]))
    summary = store.get(code).summary
    assert summary["summary"] == "Rahul Verma paid with card xxxxx, expiry xxxxx."
    assert summary["action_items"] == ["Email the ticket to xxxxx"]


async def test_ollama_model_and_context_can_be_overridden():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "{}"}})

    llm = OllamaLLM(Settings(), transport=httpx.MockTransport(handler), model="qwen2.5:3b", num_ctx=4096)
    await llm.json("s", "u", {"type": "object"})
    assert seen["model"] == "qwen2.5:3b"
    assert seen["options"]["num_ctx"] == 4096 and seen["options"]["num_gpu"] == 0
