"""Call analysis: PII masking (instant, by pattern) and the end-of-call summary (by a language model).

- mask_text() hides long digit runs (card, phone, account numbers) and emails as each line is finalised,
  so they are never shown or stored in clear.
- analyse_session() runs once a call stops: the model returns a summary, the caller's sentiment and a list
  of remaining sensitive values (expiry dates, CVVs, addresses, numbers spelled out in words), which are
  then masked in the saved transcript. People's names are never masked.
  A 1-5 "quality" score is switched off (commented out below): without a rubric it contradicted the
  sentiment, e.g. a friendly chat with nothing to resolve scored 1/5.
"""
import logging
import re
import time

from app.llm import LlmChain
from app.sessions import SessionStore

logger = logging.getLogger("voicebridge")

MASK = "xxxxx"
# 7+ digits, optionally separated by single spaces or dashes: card, phone, account and ID numbers.
# Shorter numbers (prices, times, years, flight numbers) are left alone.
_LONG_NUMBER = re.compile(r"\+?\d(?:[ \-]?\d){6,}")
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
_NUMBER_WORDS = {
    "zero", "oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "double", "triple",
}
# The same read out in words ("four five three two, seven seven..."): 7+ number words in a row.
_WORD = "(?:" + "|".join(sorted(_NUMBER_WORDS)) + ")"
_SPOKEN_NUMBER = re.compile(rf"(?<!\w){_WORD}(?:[\s,.\-]+{_WORD}){{6,}}(?!\w)", re.IGNORECASE)

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "What happened in the call, 2-3 sentences"},
        "purpose": {"type": "string", "description": "Why the caller called, one sentence"},
        "action_items": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Things someone still has to do after the call, e.g. 'Email the ticket to the caller'. "
            "Not facts about what already happened. Empty if nothing is left to do",
        },
        "outcome": {"type": "string", "description": "How the call ended, one sentence"},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"], "description": "The caller's mood"},
        # Disabled (see module docstring); to re-enable, restore this and "quality" in required and below.
        # "quality": {"type": "integer", "description": "How well the call went, 1 (bad) to 5 (excellent)"},
        "pii": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Sensitive values copied exactly as they appear in the transcript, one value per item",
        },
    },
    "required": ["summary", "purpose", "action_items", "outcome", "sentiment", "pii"],  # + "quality"
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You analyse call transcripts. Each line starts with the speaker's name; the first speaker is usually the "
    "VoiceBridge user and the other party may appear as 'Computer'. Refer to people by the names shown. The call may mix English and Hindi. Reply with JSON only, in English, following "
    "the schema, and write real content for every field based only on the transcript. "
    "For 'pii' list, one value per item and copied exactly as written: emails; phone, card, account or ID "
    "numbers (also when spelled out in words); card expiry dates; CVVs; dates of birth; street addresses. "
    "Never list people's names, cities, travel dates, prices or booking references. Already masked values "
    "appear as xxxxx; do not list them. Never write sensitive values (card details, emails, phone numbers, "
    "CVVs, expiry dates) in the summary, purpose, outcome or action items."
)


def mask_text(text: str) -> str:
    return _SPOKEN_NUMBER.sub(MASK, _LONG_NUMBER.sub(MASK, _EMAIL.sub(MASK, text)))


def _looks_like_name(value: str) -> bool:
    words = value.split()
    return 0 < len(words) <= 4 and all(
        w.isalpha() and w[0].isupper() and w.lower() not in _NUMBER_WORDS for w in words
    )


def apply_pii(text: str, values: list[str], keep: list[str]) -> str:
    """Mask each listed value in text, except people's names (`keep`, or anything that looks like a name)."""
    kept = {k.casefold() for k in keep}
    for value in sorted({v.strip() for v in values if isinstance(v, str)}, key=len, reverse=True):
        if len(value) < 3 or MASK in value or not any(c.isalnum() for c in value):
            continue
        if value.casefold() in kept or any(value.casefold() in k for k in kept) or _looks_like_name(value):
            continue
        pattern = r"\s+".join(re.escape(part) for part in value.split())
        text = re.sub(rf"(?<!\w){pattern}(?!\w)", MASK, text, flags=re.IGNORECASE)
    return text


def transcript_text(session) -> str:
    """'Name: text' lines, consecutive lines of one speaker joined."""
    lines: list[list[str]] = []
    for u in session.transcript:
        if lines and lines[-1][0] == u.speaker_name:
            lines[-1][1] += " " + u.text
        else:
            lines.append([u.speaker_name, u.text])
    return "\n".join(f"{name}: {text}" for name, text in lines)


async def analyse_session(store: SessionStore, code: str, llm: LlmChain) -> None:
    session = store.get(code)
    if not session.transcript:
        return
    store.set_summary(code, {"status": "pending"})
    started = time.perf_counter()
    try:
        result, provider = await llm.json(SYSTEM_PROMPT, transcript_text(session), ANALYSIS_SCHEMA)
    except Exception:
        logger.exception("Call analysis failed for session %s", code)
        store.set_summary(code, {"status": "unavailable"})
        return

    keep = [*session.participants.values(), "Computer"]
    pii = result.get("pii") or []
    for utterance in list(session.transcript):
        masked = apply_pii(utterance.text, pii, keep)
        if masked != utterance.text:
            store.update_utterance_text(code, utterance.id, masked)

    def clean(text) -> str:
        # The model can repeat (or invent) sensitive values in its own text: mask those as well.
        return apply_pii(mask_text(str(text or "")), pii, keep)

    summary = {k: clean(result.get(k)) for k in ("summary", "purpose", "outcome")}
    summary["sentiment"] = result.get("sentiment")
    # summary["quality"] = result.get("quality")  # disabled, see module docstring
    summary["action_items"] = [
        clean(a) for a in result.get("action_items") or [] if a and str(a).strip().lower() not in ("none", "n/a")
    ]
    store.set_summary(code, {"status": "ready", "provider": provider, "created_at": time.time(), **summary})
    logger.info("Analysed session %s with %s in %.1fs", code, provider, time.perf_counter() - started)
