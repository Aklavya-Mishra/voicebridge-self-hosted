"""Names for the voices in the call audio ("Computer 2" -> "Priya"), from what people say on the call.

Clues, each a vote for one voice:
- addressed, then answered: "Priya, can you test this?" and the very next line is a call voice -> that voice is Priya;
- a reply that names the line before it: a call voice speaks, then "Will do, Vikram" -> that voice is Vikram;
- an introduction on a call voice's own line: "Hi, Priya here", "This is Balram" -> that voice (two votes).

A voice is named only while it still has its default name and was not renamed by hand: a name typed in the call
always wins and is never replaced. A name already on another voice or person is never given twice. One clear clue
names a voice when no other name competes for it; otherwise it waits for a second clue.
"""
import re
from collections import Counter
from collections.abc import Callable

from rapidfuzz import fuzz

from app.sessions import Utterance

WINDOW_S = 8.0  # a reply or an answer comes within this many seconds of the line it belongs to
NAME = r"([A-Z][a-z]{2,})"
# Capitalised words that start or end a sentence but are not names.
NOT_NAMES = {
    "okay", "yes", "yeah", "sure", "hi", "hello", "hey", "thanks", "thank", "good", "great", "so", "please", "right",
    "alright", "well", "now", "then", "also", "and", "but", "the", "this", "that", "what", "when", "where", "why",
    "how", "let", "lets", "can", "could", "will", "would", "should", "maybe", "perfect", "cool", "fine", "agreed",
    "got", "done", "noted", "everyone", "team", "guys", "all", "morning", "evening", "afternoon", "today",
    "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february",
    "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "jira",
    "slack", "teams", "zoom", "computer", "speaker", "sir", "madam", "ma'am", "bro", "dude", "one", "first",
    "second", "next", "last", "just", "only", "here", "there", "yep", "nope", "no", "not", "hmm", "uh", "um",
}
_LEAD = r"(?:(?:ok(?:ay)?|so|and|alright|right|hi|hey|hello|now|then|also|great|good|thanks|thank you)[,.!]?\s+)?"
# "Priya, ..." / "Okay Priya, ..." at the start of a line: someone is spoken to.
_ADDRESS = re.compile(rf"^\s*{_LEAD}{NAME}\s*[,:!?]", re.I)
# "Will do, Vikram" / "Thanks Sarah" / "Sure Balram": someone is answered.
_REPLY = re.compile(
    rf"^\s*(?:ok(?:ay)?|sure|yes|yeah|will do|got it|thanks|thank you|right|agreed|noted|perfect|great|good)"
    rf"[,.!]?\s+{NAME}\b",
    re.I,
)
# "Hi, Priya here" / "This is Balram" / "I'm Sarah" on the speaker's own line.
_INTRO = re.compile(rf"\b(?:this is|i am|i'm|it's|myself)\s+{NAME}\b|\b{NAME}\s+here\b", re.I)


def _name(match: re.Match | None) -> str | None:
    if match is None:
        return None
    word = next(g for g in match.groups() if g)
    if word.lower() in NOT_NAMES:
        return None
    return word[:1].upper() + word[1:].lower()


def addressed(text: str) -> str | None:
    """The person a line speaks to at its start ("Priya, can you..."), if any."""
    return _name(_ADDRESS.search(text))


def answered(text: str) -> str | None:
    """The person a line answers ("Will do, Vikram"), if any."""
    return _name(_REPLY.search(text))


def introduced(text: str) -> str | None:
    """The name a speaker gives for themselves ("Hi, Priya here"), if any."""
    return _name(_INTRO.search(text))


def name_votes(lines: list[Utterance], is_voice: Callable[[str], bool]) -> dict[str, Counter]:
    """Speaker id of each call voice -> votes for the names the conversation gives it."""
    lines = [u for u in sorted(lines, key=lambda u: u.start_ts) if u.text.strip()]
    votes: dict[str, Counter] = {}

    def vote(speaker_id: str, name: str | None, weight: int = 1) -> None:
        if name:
            votes.setdefault(speaker_id, Counter())[name] += weight

    for i, line in enumerate(lines):
        if is_voice(line.speaker_id):
            vote(line.speaker_id, introduced(line.text), 2)
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if nxt and is_voice(nxt.speaker_id) and nxt.speaker_id != line.speaker_id and nxt.start_ts - line.end_ts <= WINDOW_S:
            vote(nxt.speaker_id, addressed(line.text))
        prev = lines[i - 1] if i > 0 else None
        if prev and is_voice(prev.speaker_id) and prev.speaker_id != line.speaker_id and line.start_ts - prev.end_ts <= WINDOW_S:
            vote(prev.speaker_id, answered(line.text))
    return votes


def clashes(name: str, taken: set[str]) -> bool:
    """Whether `name` is, or sounds like (a misheard spelling: "Aquavia" for Aklavya), a name already taken."""
    n = name.lower()
    for t in taken:
        t = t.lower()
        score = fuzz.ratio(n, t)
        if n == t or score >= 75 or (score >= 55 and n[:1] == t[:1]):
            return True
    return False


def decide(votes: dict[str, Counter], unnamed: set[str], taken: set[str]) -> dict[str, str]:
    """New names for the voices in `unnamed`, never a name in `taken` (people and voices already named) and never one
    name for two voices. Strongest clues first."""
    taken = {t.lower() for t in taken}
    # A taken name is never given, so it does not compete either ("Good. Aklavya, ..." is not a vote against Priya).
    free = {sid: Counter({n: k for n, k in c.items() if not clashes(n, taken)}) for sid, c in votes.items() if sid in unnamed}
    ranked = sorted(
        ((count, sid, name, sum(c.values()) - count) for sid, c in free.items() for name, count in c.items()),
        reverse=True,
    )
    chosen: dict[str, str] = {}
    for count, sid, name, others in ranked:
        if sid in chosen or clashes(name, taken):
            continue
        if count >= 2 or (count == 1 and others == 0):
            chosen[sid] = name
            taken.add(name.lower())
    return chosen
