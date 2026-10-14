"""Domain packs: the five shipped packs load, broken packs are rejected."""
import json

import pytest

from app.nba.knowledge import DOMAINS, PACKS, PackError, load_packs, parse_pack


def minimal(**over) -> dict:
    raw = {
        "domain": "demo",
        "label": "Demo",
        "description": "d",
        "intents": [{"name": "general", "description": "x", "guidance": "g", "priority": "suggestion"}],
        "facts": [{"key": "plan", "description": "p"}],
        "phrase_rules": [{"id": "demo-1", "speaker": "customer", "phrases": ["cancel"], "hint": "h", "priority": "alert"}],
        "fact_rules": [
            {"id": "demo-2", "fact": "plan", "when": "equals", "values": ["gold"], "hint": "h", "priority": "suggestion"}
        ],
    }
    raw.update(over)
    return raw


def test_all_five_domains_load():
    assert set(DOMAINS) == {"general", "sales", "marketing", "telecom", "tech_support"}
    for pack in PACKS.values():
        assert any(i.name == "general" for i in pack.intents)
        assert pack.facts
        # General (greetings, thanks) is answered from its guidance; the topic packs have rules.
        assert pack.domain == "general" or (pack.phrase_rules and pack.fact_rules)


def test_minimal_pack_parses():
    pack = parse_pack(minimal())
    assert pack.fact_rules[0].values == ("gold",)
    assert pack.phrase_rules[0].phrases == ("cancel",)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"intents": [{"name": "other", "description": "x", "guidance": "g", "priority": "suggestion"}]}, "general"),
        ({"phrase_rules": [{"id": "r", "speaker": "customer", "phrases": ["x"], "hint": "h", "priority": "urgent"}]}, "priority"),
        ({"fact_rules": [{"id": "r", "fact": "plan", "when": "sometimes", "values": [], "hint": "h", "priority": "alert"}]}, "when"),
        ({"fact_rules": [{"id": "r", "fact": "plan", "when": "equals", "values": [], "hint": "h", "priority": "alert"}]}, "values"),
        ({"fact_rules": [{"id": "r", "fact": "nope", "when": "present", "values": [], "hint": "h", "priority": "alert"}]}, "unknown fact"),
        ({"phrase_rules": [{"id": "r", "speaker": "robot", "phrases": ["x"], "hint": "h", "priority": "alert"}]}, "speaker"),
        ({"phrase_rules": [{"id": "r", "speaker": "customer", "phrases": [], "hint": "h", "priority": "alert"}]}, "phrases"),
    ],
)
def test_bad_packs_are_rejected(change, message):
    with pytest.raises(PackError, match=message):
        parse_pack(minimal(**change))


def test_missing_field_is_rejected():
    raw = minimal()
    del raw["facts"]
    with pytest.raises(PackError, match="facts"):
        parse_pack(raw)


def test_rule_ids_must_be_unique_across_packs(tmp_path):
    for name in ("one", "two"):
        (tmp_path / f"{name}.json").write_text(json.dumps(minimal(domain=name)), encoding="utf-8")
    with pytest.raises(PackError, match="duplicate rule id"):
        load_packs(tmp_path)


def test_file_name_must_match_domain(tmp_path):
    (tmp_path / "other.json").write_text(json.dumps(minimal()), encoding="utf-8")
    with pytest.raises(PackError, match="file name"):
        load_packs(tmp_path)
