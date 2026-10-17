"""Jira tickets from the call: a spoken request opens a draft, the model titles it, the advisor creates it."""
import asyncio

import pytest

from app.config import Settings
from app.jira_client import JiraError
from app.sessions import SessionStore, Utterance
from app.tickets import TicketService, asks_for_ticket, fallback_title
from tests.test_nba_engine import FakeLLM

SETTINGS = Settings(
    jira_url="https://example.atlassian.net", jira_email="me@example.com", jira_api_token="t", jira_project_key="SCRUM"
)
ME = {"id": "u1", "name": "Aklavya"}


class FakeJira:
    def __init__(self, fail: str | None = None):
        self.fail = fail
        self.created: list[tuple[str, str]] = []

    async def create_issue(self, title, description):
        if self.fail:
            raise JiraError(self.fail)
        self.created.append((title, description))
        key = f"SCRUM-{len(self.created) + 4}"
        return {"key": key, "url": f"https://example.atlassian.net/browse/{key}"}


class FailingLLM:
    async def json(self, system, user, schema):
        raise RuntimeError("ollama down")


def make(llm=None, jira=None):
    store = SessionStore()
    code = store.create(ME).code
    events: list[tuple[str, dict]] = []

    async def emit(c, event, payload):
        events.append((event, payload))

    service = TicketService(store, SETTINGS, llm or FakeLLM({"title": "Transcription misses words", "description": "d"}),
                            jira or FakeJira(), emit)
    return service, store, code, events


async def say(service, store, code, text, speaker=("u1-call", "Computer"), ts=None):
    ts = ts if ts is not None else len(store.get(code).transcript) + 1.0
    u = Utterance(id=f"x{ts}", speaker_id=speaker[0], speaker_name=speaker[1], text=text, start_ts=ts, end_ts=ts + 1)
    store.add_utterance(code, u)
    await service.on_final(code, u)
    await service.drain()
    return u


@pytest.mark.parametrize(
    "line",
    [
        "Can you create a Jira ticket where transcription has issues?",
        "I'll raise a ticket for that.",
        "Please log a Jira for the login error",
        "Create a ticket.",
        "open a jira ticket about the dropped calls",
    ],
)
def test_requests_for_a_ticket_are_heard(line):
    assert asks_for_ticket(line, SETTINGS.nba_phrase_match) is not None


@pytest.mark.parametrize(
    "line",
    ["The tickets were sold out.", "I bought a train ticket yesterday.", "Did you get my email?", "", "ticket"],
)
def test_other_lines_are_not_requests(line):
    assert asks_for_ticket(line, SETTINGS.nba_phrase_match) is None


def test_fallback_title_is_what_follows_the_request():
    m = SETTINGS.nba_phrase_match
    assert fallback_title("Create a Jira ticket where transcription has issues.", m, "AB12CD") == "Transcription has issues"
    assert fallback_title("Please raise a ticket for the login error", m, "AB12CD") == "The login error"
    assert fallback_title("Create a ticket.", m, "AB12CD") == "Follow-up from call AB12CD"


async def test_a_request_from_either_speaker_opens_a_draft_titled_by_the_model():
    llm = FakeLLM({"title": "transcription misses words on mobile.", "description": "Words are  dropped."})
    service, store, code, events = make(llm=llm)
    await say(service, store, code, "The app keeps dropping my words.")
    await say(service, store, code, "Okay, I will create a Jira ticket where transcription has issues.",
              speaker=("u1", "Aklavya"))

    statuses = [p["status"] for e, p in events if e == "ticket_draft"]
    assert statuses == ["drafting", "ready"]
    (ticket,) = store.get(code).tickets
    assert ticket["status"] == "ready"
    assert ticket["title"] == "Transcription misses words on mobile"
    assert ticket["description"] == "Words are dropped."
    assert ticket["requested_by"] == "Aklavya"
    # The model saw the earlier line and which line asked for the ticket.
    _, prompt, _ = llm.calls[0]
    assert "Computer: The app keeps dropping my words." in prompt
    assert prompt.endswith("Ticket requested in this line: Aklavya: Okay, I will create a Jira ticket where "
                           "transcription has issues.")


async def test_lines_without_a_request_do_nothing():
    service, store, code, events = make()
    await say(service, store, code, "My internet is slow since yesterday.")
    assert events == [] and store.get(code).tickets == []


async def test_saying_it_again_while_a_draft_is_open_does_not_open_another():
    service, store, code, _ = make()
    await say(service, store, code, "Create a Jira ticket for the slow internet.")
    await say(service, store, code, "Yes, create a Jira ticket please.")
    assert len(store.get(code).tickets) == 1


async def test_a_failing_model_falls_back_to_the_request_words():
    service, store, code, _ = make(llm=FailingLLM())
    await say(service, store, code, "My card 4111 1111 1111 1111 was charged twice.")
    await say(service, store, code, "Please create a Jira ticket where transcription has issues.")
    (ticket,) = store.get(code).tickets
    assert ticket["status"] == "ready"
    assert ticket["title"] == "Transcription has issues"
    assert "Requested on the call:" in ticket["description"]
    assert "4111" not in ticket["description"]  # masked


async def test_a_slow_model_falls_back(monkeypatch):
    monkeypatch.setattr("app.tickets.DRAFT_TIMEOUT_S", 0.05)
    service, store, code, _ = make(llm=FakeLLM({"title": "Late", "description": ""}, delay=1.0))
    await say(service, store, code, "Raise a ticket about the dropped calls.")
    assert store.get(code).tickets[0]["title"] == "The dropped calls"


async def test_create_uses_the_edited_title_and_records_the_key():
    jira = FakeJira()
    service, store, code, events = make(jira=jira)
    await say(service, store, code, "Create a Jira ticket where transcription has issues.")
    ticket_id = store.get(code).tickets[0]["id"]

    ticket = await service.create(code, ticket_id, "  Edited title ", "Edited description")

    assert jira.created == [("Edited title", "Edited description")]
    assert ticket["status"] == "created" and ticket["key"] == "SCRUM-5"
    assert ticket["url"] == "https://example.atlassian.net/browse/SCRUM-5"
    assert [p["status"] for e, p in events][-2:] == ["creating", "created"]
    with pytest.raises(ValueError):
        await service.create(code, ticket_id, "Again", "")  # created once only
    # After a ticket is created a new request opens a new draft.
    await say(service, store, code, "Also create a ticket for the billing page.")
    assert [t["status"] for t in store.get(code).tickets] == ["created", "ready"]


async def test_jira_failure_keeps_the_draft_to_try_again():
    jira = FakeJira(fail="Jira answered 401: Unauthorized")
    service, store, code, _ = make(jira=jira)
    await say(service, store, code, "Create a Jira ticket for the login error.")
    ticket_id = store.get(code).tickets[0]["id"]

    ticket = await service.create(code, ticket_id, "Login error", "")
    assert ticket["status"] == "failed" and "401" in ticket["error"]

    jira.fail = None
    ticket = await service.create(code, ticket_id, "Login error", "")
    assert ticket["status"] == "created" and "error" not in ticket


async def test_empty_title_and_unknown_ticket_are_refused():
    service, store, code, _ = make()
    await say(service, store, code, "Create a ticket for the outage.")
    ticket_id = store.get(code).tickets[0]["id"]
    with pytest.raises(ValueError, match="title"):
        await service.create(code, ticket_id, "   ", "")
    with pytest.raises(ValueError, match="Unknown"):
        await service.create(code, "nope", "Title", "")


async def test_dismiss_closes_the_draft_and_allows_a_new_one():
    service, store, code, _ = make()
    await say(service, store, code, "Create a ticket for the outage.")
    ticket_id = store.get(code).tickets[0]["id"]
    assert (await service.dismiss(code, ticket_id))["status"] == "dismissed"
    await say(service, store, code, "Actually, create a Jira ticket for the outage.")
    assert [t["status"] for t in store.get(code).tickets] == ["dismissed", "ready"]


async def test_tickets_are_saved_with_the_call(tmp_path):
    db = str(tmp_path / "calls.db")
    store = SessionStore(db_path=db)
    code = store.create(ME).code

    async def emit(*args):
        pass

    service = TicketService(store, SETTINGS, FakeLLM({"title": "Outage", "description": ""}), FakeJira(), emit)
    await say(service, store, code, "Create a ticket for the outage.")
    await service.create(code, store.get(code).tickets[0]["id"], "Outage", "")

    (ticket,) = SessionStore(db_path=db).get(code).tickets
    assert ticket["status"] == "created" and ticket["key"] == "SCRUM-5"


async def test_drafting_does_not_block_the_caller():
    service, store, code, _ = make(llm=FakeLLM({"title": "T", "description": ""}, delay=0.3))
    u = Utterance(id="a", speaker_id="u1", speaker_name="Aklavya", text="Create a ticket for it.", start_ts=1, end_ts=2)
    store.add_utterance(code, u)
    started = asyncio.get_running_loop().time()
    await service.on_final(code, u)
    assert asyncio.get_running_loop().time() - started < 0.1
    assert store.get(code).tickets[0]["status"] == "drafting"
    await service.drain()


async def test_auto_create_creates_the_ticket_without_a_click():
    from dataclasses import replace

    store = SessionStore()
    code = store.create(ME).code
    jira = FakeJira()

    async def emit(c, event, payload):
        pass

    service = TicketService(store, replace(SETTINGS, jira_auto_create=True),
                            FakeLLM({"title": "OTP retry fails", "description": "d"}), jira, emit)
    await say(service, store, code, "please create a Jira ticket for the OTP retry failing")
    ticket = store.get(code).tickets[0]
    assert ticket["status"] == "created" and ticket["key"] == "SCRUM-5"
    assert jira.created == [("OTP retry fails", "d")]
