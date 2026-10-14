"""Jira tickets drafted from the call, created only when the advisor clicks Create (human in the loop).

SocketHandlers calls on_final() for every finished line of either speaker. A line that asks for a ticket
("create a Jira ticket where transcription has issues") opens a draft at once (status drafting); the local
model then writes its title and description from the last lines (status ready). The web app shows the draft
in the Assist panel, where it can be edited and created (create) or dismissed (dismiss).

Draft: {id, status, title, description, requested_by, trigger, ts, key?, url?, error?}
status: drafting -> ready -> creating -> created | failed (Create again) ; ready | failed -> dismissed
Drafts are saved on the session and go out as `ticket_draft` events through `emit(code, event, payload)`.
"""
import asyncio
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from app.analysis import mask_text
from app.config import Settings
from app.jira_client import JiraError
from app.nba.rules import normalize, phrase_end
from app.sessions import SessionStore, Utterance

logger = logging.getLogger("voicebridge")

Emit = Callable[[str, str, dict], Awaitable[None]]

# Ways of asking for a ticket, matched as whole words with the hint rules' fuzzy match. Longest first, so the
# fallback title starts after the most complete phrase.
TRIGGER_PHRASES = sorted(
    [
        "create a jira ticket", "create jira ticket", "create a jira", "create a ticket", "raise a jira ticket",
        "raise a jira", "raise a ticket", "log a jira ticket", "log a jira", "log a ticket", "open a jira ticket",
        "open a jira", "open a ticket", "file a jira", "file a ticket", "make a jira ticket", "jira ticket",
    ],
    key=len,
    reverse=True,
)
OPEN = ("drafting", "ready", "creating", "failed")  # a new request is ignored while one of these is open
CONTEXT_LINES = 8
DRAFT_TIMEOUT_S = 45.0  # includes waiting behind a hint turn on the same local model
MAX_TITLE = 80
# Words between the request and the subject: "create a ticket *for* the login error".
LEADING_FILLER = {"where", "for", "about", "that", "saying", "regarding", "on", "because", "to", "so", "and", "which"}

SYSTEM = (
    "You write Jira tickets from a live call transcript. Someone on the call asked for a ticket; the line that "
    "asked is given. Return JSON with:\n"
    "- title: a short, specific issue title in sentence case, at most 80 characters, naming the problem or task "
    "(e.g. 'Transcription misses words on mobile calls'). Never start with words like 'Create', 'Ticket' or "
    "'Jira'. No personal data such as names, phone numbers or emails.\n"
    "- description: 1 to 3 sentences on what the issue or request is and the relevant context from the call.\n"
    "Use only what the transcript says. Write in English."
)
SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "description": {"type": "string"}},
    "required": ["title", "description"],
}


def asks_for_ticket(text: str, threshold: int) -> int | None:
    """End of the request phrase in the normalized line when the line asks for a ticket, else None."""
    line = normalize(text)
    for phrase in TRIGGER_PHRASES:
        end = phrase_end(line, phrase, threshold)
        if end is not None:
            return end
    return None


def fallback_title(text: str, threshold: int, code: str) -> str:
    """Title from the words after the request ("...where transcription has issues" -> "Transcription has
    issues"), used when the model cannot write one."""
    line = normalize(text)
    end = asks_for_ticket(text, threshold)
    words = line[end:].split() if end is not None else []
    while words and words[0] in LEADING_FILLER:
        words.pop(0)
    if not words:
        return f"Follow-up from call {code}"
    return _short(" ".join(words)).capitalize()


def _short(text: str, limit: int = MAX_TITLE) -> str:
    text = " ".join(str(text or "").split()).strip(" \"'")
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:-")


def _clean_title(title: str) -> str:
    title = _short(re.sub(r"^(title|summary)\s*:\s*", "", str(title or ""), flags=re.I)).rstrip(".")
    return title[:1].upper() + title[1:]


class TicketService:
    def __init__(self, store: SessionStore, settings: Settings, llm, jira, emit: Emit):
        self._store = store
        self._settings = settings
        self._llm = llm
        self._jira = jira
        self._emit = emit
        self._tasks: set[asyncio.Task] = set()

    async def on_final(self, code: str, utterance: Utterance) -> None:
        if not utterance.text.strip() or asks_for_ticket(utterance.text, self._settings.nba_phrase_match) is None:
            return
        if any(t["status"] in OPEN for t in self._tickets(code)):
            return  # one open draft per call: saying it again does not open a second card
        draft = {
            "id": uuid.uuid4().hex[:12],
            "status": "drafting",
            "title": "",
            "description": "",
            "requested_by": utterance.speaker_name,
            "trigger": utterance.text,
            "ts": time.time(),
        }
        await self._put(code, draft)
        task = asyncio.create_task(self._draft(code, draft["id"], utterance))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """Wait for drafts being written (tests)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def _draft(self, code: str, ticket_id: str, trigger: Utterance) -> None:
        recent = self._recent(code, trigger)
        prompt = "\n".join(recent) + f"\n\nTicket requested in this line: {trigger.speaker_name}: {trigger.text}"
        try:
            reply = await asyncio.wait_for(self._llm.json(SYSTEM, prompt, SCHEMA), DRAFT_TIMEOUT_S)
            title = _clean_title(reply.get("title"))
            description = " ".join(str(reply.get("description") or "").split())
            if not title:
                raise ValueError("empty title")
        except Exception as exc:  # slow or failing model: a title from the request itself, editable anyway
            logger.warning("Ticket draft from the model failed for session %s: %s", code, exc or type(exc).__name__)
            title = fallback_title(trigger.text, self._settings.nba_phrase_match, code)
            description = "Requested on the call:\n" + "\n".join(recent[-4:])
        draft = self._find(code, ticket_id)
        if draft is None or draft["status"] != "drafting":
            return
        draft.update(title=mask_text(title), description=mask_text(description), status="ready")
        await self._put(code, draft)

    def _recent(self, code: str, trigger: Utterance) -> list[str]:
        """The last lines of the call up to the request, as "Speaker: text"."""
        transcript = self._store.get(code).transcript
        upto = [u for u in transcript if u.start_ts <= trigger.start_ts and u.text.strip()]
        return [f"{u.speaker_name}: {u.text}" for u in upto[-CONTEXT_LINES:]]

    async def create(self, code: str, ticket_id: str, title: str, description: str) -> dict:
        """Create the Jira issue for a ready (or failed) draft with the advisor's title and description."""
        draft = self._find(code, ticket_id)
        if draft is None:
            raise ValueError("Unknown ticket")
        if draft["status"] not in ("ready", "failed"):
            raise ValueError(f"Ticket is {draft['status']}")
        title = _short(title, 255)
        if not title:
            raise ValueError("Give the ticket a title")
        draft.update(title=title, description=str(description or "").strip(), status="creating")
        draft.pop("error", None)
        await self._put(code, draft)
        try:
            issue = await self._jira.create_issue(draft["title"], draft["description"])
        except JiraError as exc:
            draft.update(status="failed", error=str(exc))
        except Exception as exc:
            logger.exception("Jira ticket failed for session %s", code)
            draft.update(status="failed", error=f"Could not create the ticket: {type(exc).__name__}")
        else:
            draft.update(status="created", key=issue["key"], url=issue["url"], created_at=time.time())
            logger.info("Created Jira ticket %s for session %s", issue["key"], code)
        await self._put(code, draft)
        return draft

    async def dismiss(self, code: str, ticket_id: str) -> dict:
        draft = self._find(code, ticket_id)
        if draft is None:
            raise ValueError("Unknown ticket")
        if draft["status"] not in ("drafting", "ready", "failed"):
            raise ValueError(f"Ticket is {draft['status']}")
        draft["status"] = "dismissed"
        await self._put(code, draft)
        return draft

    def _tickets(self, code: str) -> list:
        return self._store.get(code).tickets

    def _find(self, code: str, ticket_id: str) -> dict | None:
        return next((t for t in self._tickets(code) if t["id"] == ticket_id), None)

    async def _put(self, code: str, draft: dict) -> None:
        tickets = [t for t in self._tickets(code) if t["id"] != draft["id"]] + [draft]
        tickets.sort(key=lambda t: t["ts"])
        self._store.set_tickets(code, tickets)
        await self._emit(code, "ticket_draft", {"code": code, **draft})
