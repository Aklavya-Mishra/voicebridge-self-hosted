"""Instant hints from a domain pack's rules, without a language model: a phrase the customer (or advisor)
said, or a fact that is known, equal to a value, or still missing. Each rule fires at most once per call."""
import re
import time
import unicodedata

from rapidfuzz import fuzz

from app.nba.knowledge import Pack


def normalize(text: str) -> str:
    """Lowercase ASCII words separated by single spaces: for comparing, never for display."""
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def _hint(rule, ts: float | None) -> dict:
    return {
        "id": rule.id,
        "text": rule.hint,
        "kind": "rule",
        "intent": None,
        "priority": rule.priority,
        "ts": time.time() if ts is None else ts,
    }


def _says(line: str, phrase: str, threshold: int) -> bool:
    phrase = normalize(phrase)
    # A line shorter than the phrase cannot contain it ("No." must not match "no signal").
    if not phrase or len(line) < len(phrase):
        return False
    match = fuzz.partial_ratio_alignment(phrase, line, score_cutoff=threshold)
    if match is None:
        return False
    # The matched part must start and end on whole words: "it s working since y" is not "not working since".
    start, end = match.dest_start, match.dest_end
    while start < end and line[start] == " ":
        start += 1
    while end > start and line[end - 1] == " ":
        end -= 1
    return (start == 0 or line[start - 1] == " ") and (end == len(line) or line[end] == " ")


def match_phrase_rules(
    pack: Pack, line: str, speaker: str, fired: set[str], threshold: int, ts: float | None = None
) -> list[dict]:
    text = normalize(line)
    if not text:
        return []
    return [
        _hint(rule, ts)
        for rule in pack.phrase_rules
        if rule.id not in fired and rule.speaker == speaker and any(_says(text, p, threshold) for p in rule.phrases)
    ]


def match_fact_rules(
    pack: Pack, facts: dict[str, str], customer_lines: int, fired: set[str], absent_after: int, ts: float | None = None
) -> list[dict]:
    known = {k: v for k, v in facts.items() if str(v).strip()}
    hints = []
    for rule in pack.fact_rules:
        if rule.id in fired:
            continue
        if rule.when == "present":
            hit = rule.fact in known
        elif rule.when == "absent":
            hit = rule.fact not in known and customer_lines >= absent_after
        else:  # equals
            hit = rule.fact in known and normalize(known[rule.fact]) in {normalize(v) for v in rule.values}
        if hit:
            hints.append(_hint(rule, ts))
    return hints
