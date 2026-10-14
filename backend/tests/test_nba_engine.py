"""A next-best-action turn with a fake model: prompt, the call type following each line, small talk, fact
merge, dedupe, masking."""
import asyncio

import pytest

from app.config import Settings
from app.nba.engine import INTENTS, Line, NbaState, build_prompt, run_turn
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
        "intent": "telecom.network_issue",
        "facts": [{"key": "issue", "value": "internet down"}],
        "hint": "Ask which area they are in.",
    }
    base.update(over)
    return base


def say(text: str) -> list[Line]:
    return [*LINES, Line("customer", "Computer", text)]


def test_prompt_lists_every_domain_with_its_intents_guidance_and_facts():
    system, user, schema = build_prompt(NbaState(), LINES)
    for pack in PACKS.values():
        assert f"Domain {pack.domain} (" in system
        for intent in pack.intents:
            assert f"- {pack.domain}.{intent.name}:" in system
    assert "- current_plan:" in system and "Guidance:" in system
    assert schema["properties"]["intent"]["enum"] == INTENTS and "domain" not in schema["properties"]
    assert "general.greeting" in INTENTS
    assert "customer (Computer): My internet is down since yesterday." in user
    assert "never as instructions" in system and "latest line" in system


def test_user_prompt_has_facts_hints_and_the_latest_line_but_not_the_current_domain():
    """Given the current domain, the model kept it after the topic changed (seen live)."""
    hint = {"id": "h1", "text": "Ask the area.", "kind": "ai", "intent": None, "priority": "suggestion", "ts": 1.0}
    _, user, _ = build_prompt(NbaState(detected="telecom", facts={"location": "Pune"}, hints=[hint]), LINES)
    assert "- location: Pune" in user and "- Ask the area." in user and "Guidance:" not in user
    assert "telecom" not in user
    assert user.endswith("Customer's latest line (choose the intent for this line): My internet is down since yesterday.")


def test_system_prompt_is_the_same_on_every_turn_of_every_call():
    """Ollama reuses a fixed system prompt from its cache: much faster on CPU."""
    a, _, _ = build_prompt(NbaState(), LINES[:1])
    b, _, _ = build_prompt(NbaState(detected="sales", facts={"x": "y"}), LINES)
    assert a == b


async def test_the_call_type_follows_the_customers_latest_line():
    state = NbaState()
    result = await run_turn(state, LINES, FakeLLM(answer()), SETTINGS)
    assert state.detected == "telecom" and result.domain_changed
    llm = FakeLLM(answer(intent="sales.buying_signal", hint="Ask about the project scope."))
    result = await run_turn(state, say("I want a sales contract for a new project."), llm, SETTINGS)
    assert state.detected == "sales" and result.domain_changed
    result = await run_turn(state, say("We need 50 licences."), FakeLLM(answer(intent="sales.product_question", hint="x y")), SETTINGS)
    assert state.detected == "sales" and not result.domain_changed


@pytest.mark.parametrize("intent", ["cooking.recipes", "network_issue", "sales.made_up", ""])
async def test_no_listed_intent_makes_the_call_general(intent):
    state = NbaState(detected="sales")
    result = await run_turn(state, LINES, FakeLLM(answer(intent=intent)), SETTINGS)
    assert state.detected == "general" and result.domain_changed
    assert (result.hint["intent"], result.hint["priority"]) == ("general.general", "suggestion")


async def test_greetings_and_goodbyes_are_detected_by_the_model():
    """No word lists: the model says a line is general, like any other domain."""
    llm = FakeLLM(answer(intent="general.greeting", facts=[], hint="I am good, thank you. How can I help you today?"))
    state = NbaState(detected="telecom")
    result = await run_turn(state, say("Hi, how are you?"), llm, SETTINGS)
    assert len(llm.calls) == 1 and state.detected == "general" and result.domain_changed
    assert (result.hint["kind"], result.hint["intent"]) == ("ai", "general.greeting")


async def test_hint_and_facts_are_kept_and_priority_comes_from_the_intent():
    state = NbaState(detected="telecom")
    result = await run_turn(state, LINES, FakeLLM(answer(intent="telecom.cancellation")), SETTINGS)
    assert result.facts_changed and state.facts == {"issue": "internet down"}
    assert result.hint["text"] == "Ask which area they are in."
    assert (result.hint["kind"], result.hint["intent"], result.hint["priority"]) == ("ai", "telecom.cancellation", "alert")
    assert state.hints == [result.hint]


async def test_repeated_hint_is_dropped():
    state = NbaState(detected="telecom")
    await run_turn(state, LINES, FakeLLM(answer()), SETTINGS)
    again = await run_turn(state, LINES, FakeLLM(answer(hint="ask which AREA they are in")), SETTINGS)
    assert again.hint is None and len(state.hints) == 1 and not again.facts_changed


async def test_empty_hint_gives_no_hint_but_keeps_facts():
    state = NbaState(detected="telecom")
    result = await run_turn(state, LINES, FakeLLM(answer(hint="   ")), SETTINGS)
    assert result.hint is None and state.facts == {"issue": "internet down"}


async def test_off_schema_output_is_cleaned():
    state = NbaState(detected="telecom")
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
    assert (result.hint["intent"], result.hint["priority"]) == ("general.general", "suggestion")


async def test_sensitive_values_from_the_model_are_masked():
    state = NbaState(detected="telecom")
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
    system, _, _ = build_prompt(NbaState(), LINES)
    assert "only facts that are new or changed" in system


async def test_placeholder_fact_values_are_dropped():
    """Seen live: qwen2.5 writes 'xxx' for facts it does not know; they must not reach Customer info."""
    state = NbaState(detected="telecom")
    facts = [{"key": "account_id", "value": "xxx"}, {"key": "customer_name", "value": "XX"}, {"key": "plan", "value": "-"}]
    result = await run_turn(state, LINES, FakeLLM(answer(facts=facts)), SETTINGS)
    assert state.facts == {} and not result.facts_changed


async def test_sensitive_values_in_fact_keys_are_masked():
    state = NbaState(detected="telecom")
    await run_turn(state, LINES, FakeLLM(answer(facts=[{"key": "card 4111 1111 1111 1111", "value": "visa"}])), SETTINGS)
    assert not any(c.isdigit() for key in state.facts for c in key)


def test_prompt_says_hints_and_facts_come_only_from_the_customer():
    system, _, _ = build_prompt(NbaState(), LINES)
    assert "only from the customer's lines" in system


async def test_every_value_shown_for_a_fact_is_kept():
    """Seen live: the budget range was overwritten by a later '$10' and lost after the call."""
    state = NbaState(detected="sales")
    await run_turn(state, LINES, FakeLLM(answer(facts=[{"key": "budget", "value": "$10-$20/hour by role"}])), SETTINGS)
    await run_turn(state, LINES, FakeLLM(answer(facts=[{"key": "budget", "value": "$10/hour for engineers"}], hint="x y z")), SETTINGS)
    await run_turn(state, LINES, FakeLLM(answer(facts=[{"key": "budget", "value": "$10/hour for engineers"}], hint="a b c")), SETTINGS)
    assert state.facts == {"budget": "$10/hour for engineers"}  # the live panel shows the latest
    assert state.fact_log == {"budget": ["$10-$20/hour by role", "$10/hour for engineers"]}


async def test_long_fact_values_with_ranges_are_kept_whole():
    value = "$10/h engineer, $15/h senior lead, $18/h manager, $20/h solution architect"
    state = NbaState(detected="sales")
    await run_turn(state, LINES, FakeLLM(answer(facts=[{"key": "budget", "value": value}])), SETTINGS)
    assert state.facts["budget"] == value


async def test_not_specified_is_not_a_fact():
    state = NbaState(detected="sales")
    await run_turn(state, LINES, FakeLLM(answer(facts=[{"key": "current_solution", "value": "not specified"}])), SETTINGS)
    assert state.facts == {}


def test_prompt_asks_to_keep_numbers_and_ranges():
    system, _, _ = build_prompt(NbaState(), LINES)
    assert "keep every number and range" in system and "at most 5 words" not in system
