"""Slack recap: built from the ready summary, posted once as the user, then edited; failures kept for Retry."""

from app.config import Settings
from app.recap import RecapService, build_message, match_member
from app.sessions import SessionStore, Utterance
from app.slack_client import SlackError

SETTINGS = Settings(slack_user_token="xoxp-1", slack_recap_channel="voicebridge-recaps")
ME = {"id": "u1", "name": "Aklavya"}
SUMMARY = {
    "status": "ready", "summary": "Two bugs were found in login.", "purpose": "Sprint bug triage",
    "outcome": "Both bugs logged", "action_items": ["Balram will fix the OTP retry by Friday"],
    "people": ["Aklavya", "Balram", "Priya"],
}


class FakeSlack:
    def __init__(self, fail: str | None = None, members=None):
        self.fail = fail
        self.posts: list[tuple] = []
        self.updates: list[tuple] = []
        self._members = members or []

    async def post(self, channel, text, blocks):
        if self.fail:
            raise SlackError(self.fail)
        self.posts.append((channel, text, blocks))
        return {"channel": "C1", "ts": "111.2"}

    async def update(self, channel, ts, text, blocks):
        if self.fail:
            raise SlackError(self.fail)
        self.updates.append((channel, ts, text, blocks))

    async def permalink(self, channel, ts):
        return "https://slack.example/p111"

    async def members(self):
        return self._members


def make(slack=None, lookup=None, summary=SUMMARY, kind="live"):
    store = SessionStore()
    session = store.create(ME)
    session.kind = kind
    store.add_utterance(session.code, Utterance(id="a", speaker_id="u1", speaker_name="Aklavya", text="Hi team",
                                                start_ts=1000.0, end_ts=1002.0))
    store.add_utterance(session.code, Utterance(id="b", speaker_id="u1-call", speaker_name="Computer", text="Hi",
                                                start_ts=1100.0, end_ts=1240.0))
    store.set_summary(session.code, dict(summary))
    events = []

    async def emit(code, event, payload):
        events.append((event, payload))

    slack = slack or FakeSlack()
    return RecapService(store, SETTINGS, emit, client=slack, lookup=lookup), store, session.code, slack, events


def texts(blocks):
    return "\n".join(b["text"]["text"] if "text" in b else b["elements"][0]["text"] for b in blocks)


def test_call_message_has_title_people_sections_and_tickets():
    _, store, code, _, _ = make()
    store.set_tickets(code, [{"id": "t", "status": "created", "key": "SCRUM-12", "title": "OTP retry fails",
                              "url": "https://j/browse/SCRUM-12", "ts": 1}])
    text, blocks = build_message(store.get(code), {"Balram": "U2"})
    body = texts(blocks)
    assert blocks[0]["text"]["text"] == "Call recap: Sprint bug triage"
    assert "4 min" in text
    assert "*People:* Aklavya, <@U2>, Priya" in body
    assert "*Outcome*\n• Both bugs logged" in body
    assert "• Balram will fix the OTP retry by Friday" in body
    assert "<https://j/browse/SCRUM-12|SCRUM-12>: OTP retry fails" in body


def test_empty_sections_and_no_clear_outcome_are_left_out():
    _, store, code, _, _ = make(summary={**SUMMARY, "outcome": "No clear outcome", "action_items": []})
    body = texts(build_message(store.get(code))[1])
    assert "Outcome" not in body and "Action items" not in body and "Done during" not in body


def test_meeting_minutes_sections():
    minutes = {"status": "ready", "kind": "minutes", "summary": "Weekly sync. All on track.", "attendees": ["A", "B"],
               "decisions": ["Ship Friday"], "action_items": [], "open_issues": ["Load test"]}
    _, store, code, _, _ = make(summary=minutes)
    _, blocks = build_message(store.get(code))
    assert blocks[0]["text"]["text"] == "Meeting minutes: Weekly sync."
    body = texts(blocks)
    assert "*Decisions*\n• Ship Friday" in body and "*Open issues*" in body and "Action items" not in body


def test_member_match_needs_exactly_one():
    members = [{"id": "U1", "names": ["Balram Singh", "balram"]}, {"id": "U2", "names": ["Priya Rao"]},
               {"id": "U3", "names": ["Priya Shah"]}]
    assert match_member("Balram", members) == "U1"
    assert match_member("balram singh", members) == "U1"
    assert match_member("Priya", members) is None
    assert match_member("Nobody", members) is None


async def test_first_send_posts_then_edits_same_message():
    service, store, code, slack, events = make()
    recap = await service.send(code)
    assert recap["status"] == "sent" and recap["ts"] == "111.2" and recap["permalink"]
    assert slack.posts[0][0] == "voicebridge-recaps"
    assert store.get(code).recap["channel"] == "C1"
    assert events[-1][0] == "recap"
    await service.send(code)
    assert len(slack.posts) == 1 and slack.updates[0][:2] == ("C1", "111.2")


async def test_mentions_come_from_the_lookup_token():
    lookup = FakeSlack(members=[{"id": "U9", "names": ["Balram Singh"]}])
    service, _, code, slack, _ = make(lookup=lookup)
    await service.send(code)
    assert "<@U9>" in texts(slack.posts[0][2])


async def test_failure_is_kept_and_retry_sends():
    slack = FakeSlack(fail="Slack answered: channel_not_found")
    service, store, code, _, _ = make(slack=slack)
    recap = await service.send(code)
    assert recap["status"] == "failed" and "channel_not_found" in recap["error"]
    slack.fail = None
    recap = await service.send(code)
    assert recap["status"] == "sent" and "error" not in recap and len(slack.posts) == 1


async def test_not_sent_before_ready_or_for_imports():
    service, _, code, slack, _ = make(summary={"status": "pending"})
    assert await service.send(code) is None
    service, _, code, slack2, _ = make(kind="import")
    assert await service.send(code) is None
    assert not slack.posts and not slack2.posts


async def test_slack_client_reports_slack_error_codes():
    import httpx

    from app.slack_client import SlackClient

    def handler(request):
        assert request.headers["Authorization"] == "Bearer xoxp-1"
        if request.url.path.endswith("chat.postMessage"):
            return httpx.Response(200, json={"ok": False, "error": "not_in_channel"})
        return httpx.Response(200, json={"ok": True, "permalink": "https://x"})

    client = SlackClient("xoxp-1", transport=httpx.MockTransport(handler))
    try:
        await client.post("general", "t", [])
    except SlackError as exc:
        assert "not_in_channel" in str(exc) and "xoxp" not in str(exc)
    else:
        raise AssertionError("expected SlackError")
    assert await client.permalink("C1", "1.2") == "https://x"
