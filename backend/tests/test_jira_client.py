"""Jira client: the create-issue request it sends, and the errors it reports."""
import base64
import json

import httpx
import pytest

from app.config import Settings
from app.jira_client import JiraClient, JiraError, adf

SETTINGS = Settings(
    jira_url="https://example.atlassian.net",
    jira_email="me@example.com",
    jira_api_token="secret-token",
    jira_project_key="SCRUM",
)


def client_with(handler) -> JiraClient:
    return JiraClient(SETTINGS, transport=httpx.MockTransport(handler))


async def test_creates_an_issue_and_returns_its_key_and_link():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "10005", "key": "SCRUM-5"})

    issue = await client_with(handler).create_issue("Transcription misses words", "Line one\n\nLine two")

    assert issue == {"key": "SCRUM-5", "url": "https://example.atlassian.net/browse/SCRUM-5"}
    assert seen["url"] == "https://example.atlassian.net/rest/api/3/issue"
    assert seen["auth"] == "Basic " + base64.b64encode(b"me@example.com:secret-token").decode()
    fields = seen["body"]["fields"]
    assert fields["project"] == {"key": "SCRUM"}
    assert fields["issuetype"] == {"name": "Task"}
    assert fields["summary"] == "Transcription misses words"
    assert fields["description"] == adf("Line one\nLine two")


def test_description_is_one_paragraph_per_line():
    doc = adf("First\n\n  \nSecond")
    assert doc["type"] == "doc" and doc["version"] == 1
    assert [p["content"][0]["text"] for p in doc["content"]] == ["First", "Second"]


async def test_jira_refusal_gives_a_readable_reason_without_the_token():
    def handler(request):
        return httpx.Response(400, json={"errorMessages": [], "errors": {"issuetype": "Specify a valid issue type"}})

    with pytest.raises(JiraError) as err:
        await client_with(handler).create_issue("Title", "")
    assert "400" in str(err.value) and "Specify a valid issue type" in str(err.value)
    assert "secret-token" not in str(err.value)


async def test_unreachable_jira_is_a_jira_error():
    def handler(request):
        raise httpx.ConnectError("no route")

    with pytest.raises(JiraError, match="Cannot reach Jira"):
        await client_with(handler).create_issue("Title", "")


def test_tickets_need_all_four_settings():
    assert SETTINGS.tickets_enabled
    assert not Settings(jira_url="https://x.atlassian.net", jira_email="a@b.c", jira_project_key="SCRUM").tickets_enabled
