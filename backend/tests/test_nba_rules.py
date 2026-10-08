"""Instant rule hints: phrases said by the right speaker, facts known or missing, each rule once."""
from app.nba.knowledge import parse_pack
from app.nba.rules import match_fact_rules, match_phrase_rules
from tests.test_nba_knowledge import minimal

PACK = parse_pack(
    minimal(
        phrase_rules=[
            {"id": "p-cancel", "speaker": "customer", "phrases": ["cancel my plan", "switch provider"], "hint": "Ask why.", "priority": "alert"},
            {"id": "p-signal", "speaker": "customer", "phrases": ["no signal"], "hint": "Ask where.", "priority": "alert"},
            {"id": "p-advisor", "speaker": "advisor", "phrases": ["let me check"], "hint": "Keep them informed.", "priority": "suggestion"},
        ],
        facts=[{"key": "plan", "description": "p"}, {"key": "account_id", "description": "a"}],
        fact_rules=[
            {"id": "f-gold", "fact": "plan", "when": "equals", "values": ["Gold Plus"], "hint": "Gold customer.", "priority": "suggestion"},
            {"id": "f-plan", "fact": "plan", "when": "present", "values": [], "hint": "Plan known.", "priority": "suggestion"},
            {"id": "f-account", "fact": "account_id", "when": "absent", "values": [], "hint": "Ask account.", "priority": "suggestion"},
        ],
    )
)


def ids(hints):
    return [h["id"] for h in hints]


def test_phrase_matches_inside_a_longer_line_whatever_the_case():
    hints = match_phrase_rules(PACK, "Honestly I want to CANCEL my plan today!", "customer", set(), 88, ts=5.0)
    assert hints == [{"id": "p-cancel", "text": "Ask why.", "kind": "rule", "intent": None, "priority": "alert", "ts": 5.0}]


def test_phrase_tolerates_a_small_recognition_error():
    assert ids(match_phrase_rules(PACK, "i want to swich provider", "customer", set(), 88)) == ["p-cancel"]


def test_short_line_does_not_match_a_longer_phrase():
    assert match_phrase_rules(PACK, "No.", "customer", set(), 88) == []


def test_phrase_rule_only_for_its_speaker():
    assert match_phrase_rules(PACK, "let me check that", "customer", set(), 88) == []
    assert ids(match_phrase_rules(PACK, "let me check that", "advisor", set(), 88)) == ["p-advisor"]


def test_fired_rules_do_not_fire_again():
    assert match_phrase_rules(PACK, "no signal at home", "customer", {"p-signal"}, 88) == []


def test_unrelated_line_matches_nothing():
    assert match_phrase_rules(PACK, "The weather is nice in Pune today.", "customer", set(), 88) == []


def test_present_and_equals_ignore_case_and_punctuation():
    assert ids(match_fact_rules(PACK, {"plan": "gold plus.", "account_id": "x"}, 1, set(), 4)) == ["f-gold", "f-plan"]


def test_absent_waits_for_enough_customer_lines():
    assert match_fact_rules(PACK, {}, 3, set(), 4) == []
    assert ids(match_fact_rules(PACK, {}, 4, set(), 4)) == ["f-account"]


def test_blank_fact_counts_as_absent_and_fired_rules_skip():
    assert ids(match_fact_rules(PACK, {"account_id": "  "}, 4, set(), 4)) == ["f-account"]
    assert match_fact_rules(PACK, {}, 9, {"f-account"}, 4) == []
