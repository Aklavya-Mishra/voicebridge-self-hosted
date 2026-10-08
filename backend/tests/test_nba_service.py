"""The hint service over a real session store with a fake model."""
import httpx
import pytest

from app.config import Settings
from app.nba.service import NbaService
from app.sessions import SessionStore, Utterance
from tests.test_nba_engine import FakeLLM, answer

ME = {"id": "u1", "name": "Aklavya"}
SETTINGS = Settings(nba_enabled=True, nba_timeout_s=1.0)


def make(llm, domain="telecom", store=None):
    store = store or SessionStore()
    code = store.create(ME, domain=domain).code
    events = []

    async def emit(c, event, payload):
        events.append((event, payload))

    return NbaService(store, SETTINGS, llm, emit), store, code, events


def say(store, code, uid, text, caller=True) -> Utterance:
    n = float(len(store.get(code).transcript))
    u = Utterance(uid, "u1-call" if caller else "u1", "Computer" if caller else "Aklavya", text, n, n + 0.5)
    store.add_utterance(code, u)
    return u


def named(events, name):
    return [p for e, p in events if e == name]


async def test_customer_line_gives_rule_hints_at_once_then_ai_hint_and_facts():
    service, store, code, events = make(FakeLLM(answer()))
    await service.on_final(code, say(store, code, "a", "I want to cancel my plan, no signal for days"))
    assert [h["id"] for h in named(events, "nba_hint")] == ["tel-cancel", "tel-no-signal"]  # before the model
    await service.drain(code)
    hints = named(events, "nba_hint")
    assert hints[-1]["kind"] == "ai" and hints[-1]["code"] == code
    assert named(events, "nba_facts")[-1] == {"code": code, "facts": {"issue": "internet down"}}
    assert named(events, "nba_status") == [{"code": code, "state": "thinking"}, {"code": code, "state": "idle"}]
    saved = store.get(code).nba
    assert [h["id"] for h in saved["hints"]][:2] == ["tel-cancel", "tel-no-signal"]
    assert saved["facts"] == {"issue": "internet down"}


async def test_my_own_lines_do_not_trigger():
    llm = FakeLLM(answer())
    service, store, code, events = make(llm)
    await service.on_final(code, say(store, code, "a", "cancel my plan", caller=False))
    await service.drain(code)
    assert events == [] and llm.calls == []


async def test_model_down_keeps_rule_hints_and_reports_unavailable():
    class DownLLM:
        name = "ollama"

        async def json(self, *args):
            raise httpx.ConnectError("refused")

    service, store, code, events = make(DownLLM())
    await service.on_final(code, say(store, code, "a", "I want to cancel my plan"))
    await service.drain(code)
    assert [h["id"] for h in named(events, "nba_hint")] == ["tel-cancel"]
    assert named(events, "nba_status")[-1] == {"code": code, "state": "unavailable", "reason": "local model not running"}


async def test_missing_model_says_how_to_install_it():
    class MissingLLM:
        name = "ollama"

        async def json(self, *args):
            request = httpx.Request("POST", "http://127.0.0.1:11434/api/chat")
            raise httpx.HTTPStatusError("404", request=request, response=httpx.Response(404, request=request))

    service, store, code, events = make(MissingLLM())
    await service.on_final(code, say(store, code, "a", "hello"))
    await service.drain(code)
    assert named(events, "nba_status")[-1] == {
        "code": code, "state": "unavailable", "reason": f"model not installed: ollama pull {SETTINGS.nba_model}",
    }


async def test_unreadable_model_answer_is_skipped():
    service, store, code, events = make(FakeLLM("not json"))
    await service.on_final(code, say(store, code, "a", "hello there"))
    await service.drain(code)
    assert named(events, "nba_status")[-1] == {"code": code, "state": "idle"}
    assert named(events, "nba_hint") == []


async def test_auto_detection_emits_the_domain_and_then_rules_apply():
    service, store, code, events = make(FakeLLM(answer(intent="general")), domain="auto")
    await service.on_final(code, say(store, code, "a", "I want to cancel my plan"))
    assert [h["id"] for h in named(events, "nba_hint")] == ["tel-cancel"]  # all packs' phrases while detecting
    await service.drain(code)
    assert named(events, "nba_domain") == [{"code": code, "domain": "telecom", "auto": True}]
    await service.on_final(code, say(store, code, "b", "there is no signal at home"))
    assert named(events, "nba_hint")[-1]["id"] == "tel-no-signal"
    await service.drain(code)
    assert store.get(code).nba["detected"] == "telecom"


async def test_set_domain_switches_and_auto_clears_detection():
    service, store, code, events = make(FakeLLM(answer()), domain="auto")
    await service.set_domain(code, "sales")
    assert store.get(code).domain == "sales"
    assert named(events, "nba_domain")[-1] == {"code": code, "domain": "sales", "auto": False}
    await service.set_domain(code, "auto")
    assert named(events, "nba_domain")[-1] == {"code": code, "domain": None, "auto": True}
    with pytest.raises(ValueError, match="Unknown call type"):
        await service.set_domain(code, "cooking")


async def test_missing_fact_rule_fires_after_enough_customer_lines():
    service, store, code, events = make(FakeLLM(answer(hint="")))
    for n in range(4):
        await service.on_final(code, say(store, code, str(n), f"line number {n}"))
        await service.drain(code)
    assert "tel-account-unknown" in [h["id"] for h in named(events, "nba_hint")]


async def test_restart_restores_state_so_rules_do_not_fire_twice(tmp_path):
    db = str(tmp_path / "vb.db")
    service, store, code, _ = make(FakeLLM(answer()), store=SessionStore(db_path=db))
    await service.on_final(code, say(store, code, "a", "I want to cancel my plan"))
    await service.drain(code)

    reopened = SessionStore(db_path=db)
    events = []

    async def emit(c, event, payload):
        events.append((event, payload))

    again = NbaService(reopened, SETTINGS, FakeLLM(answer(hint="Offer a callback once it is fixed.")), emit)
    await again.on_final(code, say(reopened, code, "b", "cancel my plan now"))
    await again.drain(code)
    assert "tel-cancel" not in [h["id"] for h in named(events, "nba_hint")]


async def test_rename_during_call_is_kept_when_hints_are_saved_later():
    llm = FakeLLM(answer(hint="Ask Computer which area."))
    service, store, code, _ = make(llm)
    await service.on_final(code, say(store, code, "a", "my internet is down"))
    await service.drain(code)
    store.rename_speaker(code, "u1-call", "Rahul")
    llm.result = answer(hint="Offer a technician visit.")
    await service.on_final(code, say(store, code, "b", "still nothing"))
    await service.drain(code)
    texts = [h["text"] for h in store.get(code).nba["hints"] if h["kind"] == "ai"]
    assert texts == ["Ask Rahul which area.", "Offer a technician visit."]


class DownLLM:
    name = "ollama"

    async def json(self, *args):
        raise httpx.ConnectError("refused")


async def test_auto_mode_still_gives_rule_hints_when_the_model_is_down():
    service, store, code, events = make(DownLLM(), domain="auto")
    await service.on_final(code, say(store, code, "a", "I want to cancel my plan"))
    await service.drain(code)
    assert [h["id"] for h in named(events, "nba_hint")] == ["tel-cancel"]


async def test_model_server_error_is_reported():
    class BrokenLLM:
        name = "ollama"

        async def json(self, *args):
            request = httpx.Request("POST", "http://127.0.0.1:11434/api/chat")
            raise httpx.HTTPStatusError("500", request=request, response=httpx.Response(500, request=request))

    service, store, code, events = make(BrokenLLM())
    await service.on_final(code, say(store, code, "a", "hello"))
    await service.drain(code)
    assert named(events, "nba_status")[-1] == {"code": code, "state": "unavailable", "reason": "local model error (HTTP 500)"}


async def test_slow_model_is_reported():
    service, store, code, events = make(FakeLLM(answer(), delay=5))
    service._settings = Settings(nba_enabled=True, nba_timeout_s=0.05)
    await service.on_final(code, say(store, code, "a", "hello"))
    await service.drain(code)
    assert named(events, "nba_status")[-1] == {"code": code, "state": "unavailable", "reason": "model too slow (over 0.05 s)"}


async def test_my_earlier_talk_is_not_sent_as_context():
    """Seen live: 8 advisor lines then 'I want to renew my sales contract' gave a hint about the advisor's topic."""
    llm = FakeLLM(answer())
    service, store, code, _ = make(llm)
    for n in range(5):
        say(store, code, f"me{n}", f"Deepgram chunking breaks, my own topic {n}", caller=False)
    say(store, code, "me5", "Let's end this conversation for now.", caller=False)
    await service.on_final(code, say(store, code, "c1", "So I want to renew my sales contract."))
    await service.drain(code)
    user = llm.calls[-1][1]
    assert "my own topic" not in user
    assert "advisor (Aklavya): Let's end this conversation for now." in user  # the line just before is kept
    assert "customer (Computer): So I want to renew my sales contract." in user


async def test_my_replies_after_the_caller_spoke_stay_in_context():
    llm = FakeLLM(answer())
    service, store, code, _ = make(llm)
    say(store, code, "c1", "My internet is down.")
    say(store, code, "me1", "Which area are you in?", caller=False)
    await service.on_final(code, say(store, code, "c2", "Baner in Pune."))
    await service.drain(code)
    user = llm.calls[-1][1]
    assert "advisor (Aklavya): Which area are you in?" in user and "customer (Computer): My internet is down." in user


async def test_fact_history_is_saved_with_the_call():
    llm = FakeLLM(answer(facts=[{"key": "budget", "value": "$10-$20/hour"}]))
    service, store, code, _ = make(llm)
    await service.on_final(code, say(store, code, "a", "my budget"))
    await service.drain(code)
    llm.result = answer(facts=[{"key": "budget", "value": "$10/hour"}], hint="Something different now.")
    await service.on_final(code, say(store, code, "b", "that's ten"))
    await service.drain(code)
    assert store.get(code).nba["fact_log"] == {"budget": ["$10-$20/hour", "$10/hour"]}
