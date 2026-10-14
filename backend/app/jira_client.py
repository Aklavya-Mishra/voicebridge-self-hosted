"""Creates Jira Cloud issues through the REST API (v3), signed in with an email and an API token."""
import httpx

from app.config import Settings

TIMEOUT_S = 15.0
MAX_SUMMARY = 255  # Jira's limit for an issue summary


class JiraError(Exception):
    """Jira refused the request or could not be reached; the message is safe to show (never the token)."""


def adf(text: str) -> dict:
    """Plain text as an Atlassian Document Format document: one paragraph per non-empty line."""
    paragraphs = [
        {"type": "paragraph", "content": [{"type": "text", "text": line}]}
        for line in str(text or "").splitlines()
        if line.strip()
    ]
    return {"type": "doc", "version": 1, "content": paragraphs}


def _reason(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = {}
    messages = list(body.get("errorMessages") or []) + [f"{k}: {v}" for k, v in (body.get("errors") or {}).items()]
    detail = "; ".join(messages) or response.reason_phrase or "no details"
    return f"Jira answered {response.status_code}: {detail}"


class JiraClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self._url = (settings.jira_url or "").rstrip("/")
        self._auth = (settings.jira_email or "", settings.jira_api_token or "")
        self._project = settings.jira_project_key
        self._issue_type = settings.jira_issue_type
        self._transport = transport

    async def create_issue(self, title: str, description: str) -> dict:
        """Create an issue; returns {"key": "SCRUM-5", "url": "<site>/browse/SCRUM-5"}."""
        fields = {
            "project": {"key": self._project},
            "issuetype": {"name": self._issue_type},
            "summary": title.strip()[:MAX_SUMMARY],
            "description": adf(description),
        }
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=TIMEOUT_S, auth=self._auth) as client:
                response = await client.post(f"{self._url}/rest/api/3/issue", json={"fields": fields})
        except httpx.HTTPError as exc:
            raise JiraError(f"Cannot reach Jira: {type(exc).__name__}") from exc
        if response.status_code >= 300:
            raise JiraError(_reason(response))
        key = response.json()["key"]
        return {"key": key, "url": f"{self._url}/browse/{key}"}
