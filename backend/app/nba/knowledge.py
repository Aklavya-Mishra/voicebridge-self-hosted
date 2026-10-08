"""Domain packs for live next-best-action hints: per domain, the customer intents with guidance for the
advisor, the facts worth collecting, and instant rules (phrases said, facts known or missing).

One JSON file per domain in app/nba/knowledge/, named after its domain. Loaded and checked once at import;
a broken pack stops the backend with a clear error.
"""
import json
from dataclasses import dataclass
from pathlib import Path

PACK_DIR = Path(__file__).parent / "knowledge"
PRIORITIES = ("alert", "suggestion")
WHEN = ("present", "absent", "equals")
SPEAKERS = ("customer", "advisor")


class PackError(ValueError):
    pass


@dataclass(frozen=True)
class Intent:
    name: str
    description: str
    guidance: str
    priority: str


@dataclass(frozen=True)
class Fact:
    key: str
    description: str


@dataclass(frozen=True)
class PhraseRule:
    id: str
    speaker: str
    phrases: tuple[str, ...]
    hint: str
    priority: str


@dataclass(frozen=True)
class FactRule:
    id: str
    fact: str
    when: str
    values: tuple[str, ...]
    hint: str
    priority: str


@dataclass(frozen=True)
class Pack:
    domain: str
    label: str
    description: str
    intents: tuple[Intent, ...]
    facts: tuple[Fact, ...]
    phrase_rules: tuple[PhraseRule, ...]
    fact_rules: tuple[FactRule, ...]


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise PackError(message)


def _require(raw, keys, where: str) -> None:
    _check(isinstance(raw, dict), f"{where}: must be an object")
    missing = [k for k in keys if k not in raw]
    _check(not missing, f"{where}: missing {', '.join(missing)}")


def _items(raw_list, keys, where: str):
    _check(isinstance(raw_list, list), f"{where}: must be a list")
    for n, item in enumerate(raw_list):
        _require(item, keys, f"{where} {n}")
        yield item


def parse_pack(raw: dict, where: str = "pack") -> Pack:
    _require(raw, ("domain", "label", "description", "intents", "facts", "phrase_rules", "fact_rules"), where)
    intents = tuple(
        Intent(i["name"], i["description"], i["guidance"], i["priority"])
        for i in _items(raw["intents"], ("name", "description", "guidance", "priority"), f"{where} intent")
    )
    facts = tuple(Fact(f["key"], f["description"]) for f in _items(raw["facts"], ("key", "description"), f"{where} fact"))
    phrase_rules = tuple(
        PhraseRule(r["id"], r["speaker"], tuple(r["phrases"]), r["hint"], r["priority"])
        for r in _items(raw["phrase_rules"], ("id", "speaker", "phrases", "hint", "priority"), f"{where} phrase rule")
    )
    fact_rules = tuple(
        FactRule(r["id"], r["fact"], r["when"], tuple(r["values"]), r["hint"], r["priority"])
        for r in _items(raw["fact_rules"], ("id", "fact", "when", "values", "hint", "priority"), f"{where} fact rule")
    )

    names = [i.name for i in intents]
    keys = {f.key for f in facts}
    _check("general" in names, f"{where}: needs a 'general' intent")
    _check(len(set(names)) == len(names), f"{where}: duplicate intent name")
    _check(len(keys) == len(facts), f"{where}: duplicate fact key")
    for item in (*intents, *phrase_rules, *fact_rules):
        _check(item.priority in PRIORITIES, f"{where}: unknown priority '{item.priority}'")
    for rule in phrase_rules:
        _check(rule.speaker in SPEAKERS, f"{where} {rule.id}: unknown speaker '{rule.speaker}'")
        _check(bool(rule.phrases), f"{where} {rule.id}: no phrases")
    for rule in fact_rules:
        _check(rule.when in WHEN, f"{where} {rule.id}: unknown when '{rule.when}'")
        _check(rule.fact in keys, f"{where} {rule.id}: unknown fact '{rule.fact}'")
        _check(rule.when != "equals" or bool(rule.values), f"{where} {rule.id}: equals needs values")
    return Pack(raw["domain"], raw["label"], raw["description"], intents, facts, phrase_rules, fact_rules)


def load_packs(directory: Path = PACK_DIR) -> dict[str, Pack]:
    packs: dict[str, Pack] = {}
    rule_ids: set[str] = set()
    for path in sorted(directory.glob("*.json")):
        pack = parse_pack(json.loads(path.read_text(encoding="utf-8")), path.name)
        _check(pack.domain == path.stem, f"{path.name}: file name must match domain '{pack.domain}'")
        for rule in (*pack.phrase_rules, *pack.fact_rules):
            _check(rule.id not in rule_ids, f"{path.name}: duplicate rule id '{rule.id}'")
            rule_ids.add(rule.id)
        packs[pack.domain] = pack
    return packs


PACKS = load_packs()
DOMAINS = tuple(PACKS)
