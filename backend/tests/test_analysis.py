"""End-of-call analysis: summary + PII masking, and the LLM provider chain (Gemini > Anthropic > local)."""
import asyncio
import json

import httpx
import pytest

from app.analysis import ANALYSIS_SCHEMA, SENTIMENT_SCHEMA, analyse_session, apply_pii, mask_text
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
    """Answers the summary request with `result` and the sentiment request with `sentiment`."""

    def __init__(self, name, result=None, fail=False, sentiment="positive", sentiment_fails=False):
        self.name = name
        self.calls = 0
        self._result = result
        self._fail = fail
        self._sentiment = sentiment
        self._sentiment_fails = sentiment_fails
        self.sentiment_input = None

    async def json(self, system, user, schema):
        self.calls += 1
        if self._fail:
            raise RuntimeError(f"{self.name} down")
        if schema is SENTIMENT_SCHEMA:
            self.sentiment_input = user
            if self._sentiment_fails:
                raise RuntimeError("sentiment down")
            return {"sentiment": self._sentiment}
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
    "action_items": [{"owner": "", "task": "Email the ticket"}],
    "ticket_requests": [],
    "outcome": "Booked",
    "people": ["Rahul"],
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
    from app.analysis import NEXT_STEPS_SCHEMA

    assert set(ANALYSIS_SCHEMA["required"]) | set(NEXT_STEPS_SCHEMA["required"]) == set(ANALYSIS)
    assert ANALYSIS_SCHEMA["additionalProperties"] is False and NEXT_STEPS_SCHEMA["additionalProperties"] is False


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
        action_items=[{"owner": "Aklavya", "task": "Email the ticket to rahul@x.com"}],
    )
    await analyse_session(store, code, LlmChain([FakeLLM("ollama", result=leaky)]))
    summary = store.get(code).summary
    assert summary["summary"] == "Rahul Verma paid with card xxxxx, expiry xxxxx."
    assert summary["action_items"] == ["Aklavya: Email the ticket to xxxxx."]


async def test_ollama_model_and_context_can_be_overridden():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "{}"}})

    llm = OllamaLLM(Settings(), transport=httpx.MockTransport(handler), model="qwen2.5:3b", num_ctx=4096)
    await llm.json("s", "u", {"type": "object"})
    assert seen["model"] == "qwen2.5:3b"
    assert seen["options"]["num_ctx"] == 4096 and seen["options"]["num_gpu"] == 0


async def test_ollama_keeps_the_model_loaded_and_can_warm_a_prompt():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "{}"}})

    llm = OllamaLLM(Settings(ollama_keep_alive="45m"), transport=httpx.MockTransport(handler), num_ctx=4096)
    await llm.json("s", "u", {"type": "object"})
    await llm.warm("the fixed prompt")
    assert [body["keep_alive"] for body in seen] == ["45m", "45m"]
    warm = seen[1]
    assert warm["messages"][0] == {"role": "system", "content": "the fixed prompt"}
    assert warm["options"]["num_predict"] == 1 and warm["options"]["num_ctx"] == 4096  # same context: no reload
    assert "format" not in warm


async def test_a_sentiment_word_is_not_an_outcome(tmp_path):
    """Seen live: qwen2.5 wrote outcome 'Positive' while the sentiment was neutral, and the screens disagreed."""
    store, code = seeded_store(tmp_path)
    await analyse_session(store, code, LlmChain([FakeLLM("ollama", result=dict(ANALYSIS, outcome="Positive."))]))
    assert store.get(code).summary["outcome"] == "No clear outcome"


def test_schema_says_the_outcome_is_a_sentence_not_a_sentiment():
    assert "never a sentiment word" in ANALYSIS_SCHEMA["properties"]["outcome"]["description"]


async def test_sentiment_is_rated_from_the_customers_lines_only(tmp_path):
    """Seen live: given the whole call, the advisor's calm lines averaged a caller's threat to switch provider
    down to neutral."""
    store, code = seeded_store(tmp_path)
    llm = FakeLLM("ollama", result=ANALYSIS, sentiment="negative")
    await analyse_session(store, code, LlmChain([llm]))
    # The call audio, without the advisor, and already masked.
    assert llm.sentiment_input == "Rahul Verma here, card expiry xxxxx."
    assert store.get(code).summary["sentiment"] == "negative"


async def test_without_call_audio_the_customer_is_whoever_did_not_start_the_call(tmp_path):
    """A call between two VoiceBridge users, or an import: the other person is the customer."""
    store = SessionStore(db_path=str(tmp_path / "vb.db"))
    code = store.create(ME).code
    lines = [("u1", "Aklavya", "How can I help?"), ("u2", "Ben", "My bill is wrong again!")]
    for i, (sid, name, text) in enumerate(lines):
        store.add_utterance(code, Utterance(str(i), sid, name, text, float(i), i + 0.5))
    llm = FakeLLM("ollama", result=ANALYSIS)
    await analyse_session(store, code, LlmChain([llm]))
    assert llm.sentiment_input == "My bill is wrong again!"


async def test_no_sentiment_when_the_customer_never_spoke(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "vb.db"))
    code = store.create(ME).code
    store.add_utterance(code, Utterance("0", "u1", "Aklavya", "Testing the mic, one two.", 0.0, 0.5))
    llm = FakeLLM("ollama", result=ANALYSIS)
    await analyse_session(store, code, LlmChain([llm]))
    assert llm.calls == 2 and llm.sentiment_input is None  # summary and next steps; nobody to judge: no sentiment
    assert store.get(code).summary["sentiment"] is None


async def test_a_failed_sentiment_keeps_the_summary(tmp_path):
    store, code = seeded_store(tmp_path)
    await analyse_session(store, code, LlmChain([FakeLLM("ollama", result=ANALYSIS, sentiment_fails=True)]))
    summary = store.get(code).summary
    assert summary["status"] == "ready" and summary["sentiment"] is None


def test_next_steps_cover_promises_from_either_person():
    """Seen live: 'I'll email the details' and 'follow-up by next Wednesday' gave no next steps."""
    from app.analysis import NEXT_STEPS_PROMPT, NEXT_STEPS_SCHEMA

    items = NEXT_STEPS_SCHEMA["properties"]["action_items"]
    assert "someone on the call" in items["description"] and "follow-up" in items["description"]
    assert "Keep every commitment" in items["description"] and "One item per task" in items["description"]
    assert "never guess" in items["items"]["properties"]["owner"]["description"]
    assert "either person" in NEXT_STEPS_PROMPT


async def test_next_steps_are_asked_on_their_own_and_a_failure_keeps_the_summary(tmp_path):
    """Asked within the summary, qwen2.5:7b dropped one of two tasks given to one person in 3 of 5 runs."""
    from app.analysis import NEXT_STEPS_SCHEMA

    class StepsFail(FakeLLM):
        async def json(self, system, user, schema):
            if schema is NEXT_STEPS_SCHEMA:
                raise RuntimeError("next steps down")
            return await super().json(system, user, schema)

    store, code = seeded_store(tmp_path)
    await analyse_session(store, code, LlmChain([StepsFail("ollama", result=dict(ANALYSIS, pii=[]))]))
    summary = store.get(code).summary
    assert summary["status"] == "ready" and summary["summary"] == "Rahul booked a flight." and summary["action_items"] == []


def created_ticket(key="SCRUM-7", title="Invoice download crashes on Android after update", status="created"):
    return {"id": key, "status": status, "title": title, "description": "", "requested_by": "Computer",
            "trigger": "raise a Jira ticket", "ts": 1.0, "key": key if status == "created" else None}


class RecordingLLM(FakeLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summary_input = None

    async def json(self, system, user, schema):
        if schema is ANALYSIS_SCHEMA:
            self.summary_input = user
        return await super().json(system, user, schema)


async def test_created_ticket_is_done_not_a_next_step(tmp_path):
    store, code = seeded_store(tmp_path)
    store.set_tickets(code, [created_ticket(), created_ticket("x", "Dismissed one", status="dismissed")])
    result = dict(ANALYSIS, outcome="No clear outcome", pii=[], action_items=[
        {"owner": "Balram", "task": "Fix the invoice download crash by Friday"},
        {"owner": "", "task": "Call the customer back on Friday"},
    ], ticket_requests=["Create a Jira ticket for the invoice download crash on Android"])
    llm = RecordingLLM("ollama", result=result)
    await analyse_session(store, code, LlmChain([llm]))

    summary = store.get(code).summary
    assert summary["action_items"] == ["Balram: Fix the invoice download crash by Friday.", "Call the customer back on Friday."]
    assert summary["outcome"] == "Jira ticket SCRUM-7 created"


async def test_the_model_is_not_told_the_ticket_titles(tmp_path):
    """Seen live: told 'Add pagination to orders API' was created, qwen2.5 dropped 'Balram implements the pagination'."""
    store, code = seeded_store(tmp_path)
    store.set_tickets(code, [created_ticket()])
    llm = RecordingLLM("ollama", result=dict(ANALYSIS, pii=[]))
    await analyse_session(store, code, LlmChain([llm]))
    assert "Invoice download crashes" not in llm.summary_input and "SCRUM-7" not in llm.summary_input


async def test_without_a_created_ticket_the_request_is_a_next_step(tmp_path):
    store, code = seeded_store(tmp_path)
    store.set_tickets(code, [created_ticket(status="failed")])
    llm = RecordingLLM("ollama", result=dict(ANALYSIS, pii=[], ticket_requests=["Raise a Jira ticket for the crash"]))
    await analyse_session(store, code, LlmChain([llm]))
    assert store.get(code).summary["action_items"] == ["Email the ticket.", "Raise a Jira ticket for the crash."]
    assert store.get(code).summary["outcome"] == "Booked"


def test_next_steps_read_one_sentence_per_line():
    from app.analysis import sentence_lines

    class S:
        transcript = [
            Utterance("a", "u1", "Aklavya", "Create a Jira ticket for it. Sara, write the design by Tuesday? Balram,  "
                      "you implement it!", 1, 2),
            Utterance("b", "u1-call", "Sara", "Sure.", 3, 4),
        ]

    assert sentence_lines(S()) == (
        "Aklavya: Create a Jira ticket for it.\nAklavya: Sara, write the design by Tuesday?\n"
        "Aklavya: Balram,  you implement it!\nSara: Sure."
    )


def test_one_line_per_person_with_their_tasks_in_order():
    from app.analysis import tasks_by_person

    items = [
        {"owner": "Balram", "task": "fix the coupon bug by Wednesday evening."},
        {"owner": "Priya", "task": "Retest both bugs on Thursday"},
        {"owner": "balram", "task": "Implement the pagination on Thursday"},
        {"owner": "", "task": "Follow-up next week"},
        {"owner": "", "task": "Send the deck"},
        {"owner": "Priya", "task": "none"},
        "Plain text task",
    ]
    assert tasks_by_person(items) == [
        "Balram: Fix the coupon bug by Wednesday evening. Implement the pagination on Thursday.",
        "Priya: Retest both bugs on Thursday.",
        "Follow-up next week.",
        "Send the deck.",
        "Plain text task.",
    ]


def test_voice_labels_are_not_people():
    from app.analysis import people_on_call

    class S:
        transcript = [Utterance("a", "u1", "Aklavya", "hi", 1, 2), Utterance("b", "u1-call2", "Computer 2", "yo", 3, 4)]

    assert people_on_call(S(), ["Computer 3", "Speaker 1", "Vikram", "aklavya"]) == ["Aklavya", "Vikram"]


def test_a_next_step_that_only_mentions_the_ticket_stays():
    from app.analysis import _TICKET_ITEM

    assert _TICKET_ITEM.search("Aklavya to create a Jira ticket for the export timeout")
    assert _TICKET_ITEM.search("Raise a ticket for the login bug")
    assert not _TICKET_ITEM.search("Aklavya will add Ramesh's findings to the JIRA ticket")
    assert not _TICKET_ITEM.search("Ramesh to profile the export query")
