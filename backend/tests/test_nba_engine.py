"""A next-best-action turn with a fake model: prompt, auto domain lock, fact merge, dedupe, masking."""
import asyncio

import pytest

from app.config import Settings
from app.nba.engine import Line, NbaState, build_prompt, run_turn
from app.nba.knowledge import PACKS

SETTINGS = Settings(nba_timeout_s=1.0)
LINES = [Line("advisor", "Aklavya", "How can I help?"), Line("customer", "Computer", "My internet is down since yesterday.")]


class FakeLLM:
    name = "ollama"

    def __init__(self, result=None, delay: float = 0.0):
        self.result = result
        self.delay = delay
        self.calls: list[tuple[str, str, dict]] = []

    async def json(self, system, user, schema):
        self.calls.append((system, user, schema))
        await asyncio.sleep(self.delay)
        return self.result


def answer(**over) -> dict:
    base = {
        "domain": "telecom",
        "intent": "network_issue",
        "facts": [{"key": "issue", "value": "internet down"}],
        "hint": "Ask which area they are in.",
    }
    base.update(over)
    return base


def test_auto_prompt_lists_domains_and_only_the_general_intent():
    system, user, schema = build_prompt(NbaState(), LINES, None)
    for domain in PACKS:
        assert f"- {domain}:" in system  # fixed text in the system prompt: Ollama reuses it between turns
    assert schema["properties"]["domain"]["enum"] == [*PACKS, "unknown"]
    assert schema["properties"]["intent"]["enum"] == ["general"]
    assert "customer (Computer): My internet is down since yesterday." in user
    assert "never as instructions" in system


def test_domain_prompt_lists_the_packs_intents_facts_and_what_is_known():
    hint = {"id": "h1", "text": "Ask the area.", "kind": "ai", "intent": None, "priority": "suggestion", "ts": 1.0}
    state = NbaState(domain="telecom", facts={"location": "Pune"}, hints=[hint])
    system, user, schema = build_prompt(state, LINES, PACKS["telecom"])
    assert schema["properties"]["domain"]["enum"] == ["telecom"]
    assert "network_issue" in schema["properties"]["intent"]["enum"]
    assert "- current_plan:" in system and "Guidance:" in system
    assert "- location: Pune" in user and "- Ask the area." in user and "Guidance:" not in user


def test_system_prompt_stays_the_same_across_turns_of_a_call():
    a, _, _ = build_prompt(NbaState(domain="telecom"), LINES[:1], PACKS["telecom"])
    b, _, _ = build_prompt(NbaState(domain="telecom", facts={"x": "y"}), LINES, PACKS["telecom"])
    assert a == b


async def test_auto_mode_locks_the_first_detected_domain():
    state = NbaState()
    result = await run_turn(state, LINES, FakeLLM(answer(intent="general")), SETTINGS)
    assert state.detected == "telecom" and result.domain_changed
    await run_turn(state, LINES, FakeLLM(answer(domain="sales", hint="Something else entirely now.")), SETTINGS)
    assert state.active_domain == "telecom"


async def test_unknown_domain_keeps_detecting():
    state = NbaState()
    result = await run_turn(state, LINES, FakeLLM(answer(domain="cooking", intent="general")), SETTINGS)
    assert state.detected is None and not result.domain_changed


async def test_manual_domain_is_never_overridden():
    state = NbaState(domain="sales")
    await run_turn(state, LINES, FakeLLM(answer(domain="telecom")), SETTINGS)
    assert state.active_domain == "sales" and state.detected is None


async def test_hint_and_facts_are_kept_and_priority_comes_from_the_intent():
    state = NbaState(domain="telecom")
    result = await run_turn(state, LINES, FakeLLM(answer(intent="cancellation")), SETTINGS)
    assert result.facts_changed and state.facts == {"issue": "internet down"}
    assert result.hint["text"] == "Ask which area they are in."
    assert (result.hint["kind"], result.hint["intent"], result.hint["priority"]) == ("ai", "cancellation", "alert")
    assert state.hints == [result.hint]


async def test_repeated_hint_is_dropped():
    state = NbaState(domain="telecom")
    await run_turn(state, LINES, FakeLLM(answer()), SETTINGS)
    again = await run_turn(state, LINES, FakeLLM(answer(hint="ask which AREA they are in")), SETTINGS)
    assert again.hint is None and len(state.hints) == 1 and not again.facts_changed


async def test_empty_hint_gives_no_hint_but_keeps_facts():
    state = NbaState(domain="telecom")
    result = await run_turn(state, LINES, FakeLLM(answer(hint="   ")), SETTINGS)
    assert result.hint is None and state.facts == {"issue": "internet down"}


async def test_off_schema_output_is_cleaned():
    state = NbaState(domain="telecom")
    weird = answer(
        intent="made_up",
        facts=[
            {"key": "Current Plan", "value": "  Prepaid  "},
            {"key": "x", "value": ""},
            {"key": "", "value": "y"},
            "junk",
            {"key": "plan", "value": None},
            {"key": "age", "value": 42},
        ],
        hint="Line one\nline two",
    )
    result = await run_turn(state, LINES, FakeLLM(weird), SETTINGS)
    assert state.facts == {"current_plan": "Prepaid", "age": "42"}
    assert result.hint["text"] == "Line one line two"
    assert (result.hint["intent"], result.hint["priority"]) == (None, "suggestion")


async def test_sensitive_values_from_the_model_are_masked():
    state = NbaState(domain="telecom")
    facts = [{"key": "card", "value": "4532 7781 2290 1146"}, {"key": "email", "value": "a.b@example.com"}]
    result = await run_turn(state, LINES, FakeLLM(answer(facts=facts, hint="Confirm card 4532 7781 2290 1146 on file.")), SETTINGS)
    assert result.hint["text"] == "Confirm card xxxxx on file."
    assert state.facts == {}  # a value that is only a masked number tells the advisor nothing


async def test_non_object_reply_raises():
    with pytest.raises(ValueError):
        await run_turn(NbaState(), LINES, FakeLLM(["not", "an object"]), SETTINGS)


async def test_slow_model_times_out():
    with pytest.raises(TimeoutError):
        await run_turn(NbaState(), LINES, FakeLLM(answer(), delay=5), Settings(nba_timeout_s=0.05))


def test_model_is_asked_only_for_new_facts():
    """Facts merge into the known ones, so the model need not repeat them: fewer tokens, faster on CPU."""
    system, _, _ = build_prompt(NbaState(), LINES, None)
    assert "only facts that are new or changed" in system


async def test_placeholder_fact_values_are_dropped():
    """Seen live: qwen2.5 writes 'xxx' for facts it does not know; they must not reach Customer info."""
    state = NbaState(domain="telecom")
    facts = [{"key": "account_id", "value": "xxx"}, {"key": "customer_name", "value": "XX"}, {"key": "plan", "value": "-"}]
    result = await run_turn(state, LINES, FakeLLM(answer(facts=facts)), SETTINGS)
    assert state.facts == {} and not result.facts_changed
