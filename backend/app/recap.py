"""Slack recap of a call: once the summary is ready it is posted to the recap channel as the user, then kept up to
date (a ticket created later, a speaker renamed) by editing the same message.

The message is built from the session's summary only (already masked); nothing from the raw transcript is posted.
session.recap: {status: sent | failed, channel, ts, permalink, error}.
"""
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime

from app.analysis import created_tickets
from app.config import Settings
from app.sessions import SessionStore
from app.slack_client import SlackClient, SlackError

logger = logging.getLogger("voicebridge")

Emit = Callable[[str, str, dict], Awaitable[None]]
MEMBERS_TTL_S = 600.0
MAX_HEADER = 150  # Slack's limit for a header block


def _first_sentence(text: str) -> str:
    text = " ".join(str(text or "").split())
    end = text.find(". ")
    return text[: end + 1] if end > 0 else text


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def match_member(name: str, members: list[dict]) -> str | None:
    """The Slack id of the one member whose real or display name is `name` (full name or first name)."""
    wanted = name.strip().lower()
    if not wanted:
        return None
    ids = {
        m["id"]
        for m in members
        for n in m["names"]
        if n.strip().lower() == wanted or n.strip().lower().split(" ")[0] == wanted
    }
    return ids.pop() if len(ids) == 1 else None


def _bullets(items: list[str]) -> str:
    return "\n".join(f"• {i}" for i in items)


def build_message(session, mentions: dict[str, str] | None = None) -> tuple[str, list]:
    """(notification text, Block Kit blocks) for a session with a ready summary."""
    mentions = mentions or {}
    s = session.summary or {}
    meeting = s.get("kind") == "minutes"
    if meeting:
        title = "Meeting minutes: " + _first_sentence(s.get("summary", ""))
        people = list(s.get("attendees") or [])
    else:
        title = "Call recap: " + (s.get("purpose") or "Call")
        people = list(s.get("people") or [])
    title = _short(title, MAX_HEADER)
    when = datetime.fromtimestamp(session.created_at).strftime("%d %b %Y, %H:%M")
    starts = [u.start_ts for u in session.transcript]
    ends = [u.end_ts for u in session.transcript]
    minutes = round((max(ends) - min(starts)) / 60) if starts else 0
    context = f"{when} · {max(minutes, 1)} min"
    names = ", ".join(f"<@{mentions[p]}>" if p in mentions else p for p in people)

    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text", "text": title}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": context}]},
    ]
    if names:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*People:* {names}"}})
    if s.get("summary"):
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": s["summary"]}})
    if meeting:
        sections = [("Decisions", s.get("decisions")), ("Action items", s.get("action_items")),
                    ("Open issues", s.get("open_issues"))]
    else:
        outcome = s.get("outcome")
        sections = [("Outcome", [outcome] if outcome and outcome != "No clear outcome" else []),
                    ("Action items", s.get("action_items"))]
    for label, items in sections:
        if items:
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*{label}*\n{_bullets(items)}"}})
    tickets = created_tickets(session)
    if tickets:
        lines = [f"<{t['url']}|{t['key']}>: {t['title']}" for t in tickets if t.get("url")]
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "*Done during the call*\n" + _bullets(lines)}})
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": "Posted by VoiceBridge"}]})
    return f"{title} ({context})", blocks


class RecapService:
    def __init__(self, store: SessionStore, settings: Settings, emit: Emit, client: SlackClient | None = None,
                 lookup: SlackClient | None = None):
        self._store = store
        self._channel = settings.slack_recap_channel
        self._emit = emit
        self._client = client or SlackClient(settings.slack_user_token or "")
        # Looking people up needs the bot token (users:read); without it names stay plain text.
        self._lookup = lookup or (SlackClient(settings.slack_bot_token) if settings.slack_bot_token else None)
        self._members: tuple[float, list[dict]] | None = None

    async def _mentions(self, names: list[str]) -> dict[str, str]:
        if self._lookup is None:
            return {}
        if self._members is None or time.monotonic() - self._members[0] > MEMBERS_TTL_S:
            try:
                self._members = (time.monotonic(), await self._lookup.members())
            except SlackError as exc:
                logger.warning("Slack people lookup failed: %s", exc)
                return {}
        found = {n: match_member(n, self._members[1]) for n in names}
        return {n: i for n, i in found.items() if i}

    async def send(self, code: str) -> dict | None:
        """Post the recap, or edit the one already posted. Never raises: a failure is kept on the session."""
        session = self._store.get(code)
        summary = session.summary or {}
        if session.kind != "live" or summary.get("status") != "ready":
            return None
        names = list(summary.get("people") or summary.get("attendees") or [])
        text, blocks = build_message(session, await self._mentions(names))
        previous = session.recap or {}
        try:
            if previous.get("ts"):
                await self._client.update(previous["channel"], previous["ts"], text, blocks)
                recap = {**previous, "status": "sent"}
                recap.pop("error", None)
            else:
                posted = await self._client.post(self._channel, text, blocks)
                link = await self._client.permalink(posted["channel"], posted["ts"])
                recap = {"status": "sent", **posted, "permalink": link}
            recap["name"] = self._channel
            logger.info("Slack recap %s for session %s", "updated" if previous.get("ts") else "posted", code)
        except Exception as exc:
            reason = str(exc) if isinstance(exc, SlackError) else f"Could not send the recap: {type(exc).__name__}"
            logger.warning("Slack recap failed for session %s: %s", code, reason)
            recap = {**previous, "status": "failed", "error": reason, "name": self._channel}
        self._store.set_recap(code, recap)
        await self._emit(code, "recap", {"code": code, **recap})
        return recap
