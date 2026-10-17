"""Posts and edits Slack messages through the Web API, signed in with a user token (so they show as that person)."""
import httpx

TIMEOUT_S = 10.0
API = "https://slack.com/api"


class SlackError(Exception):
    """Slack refused the request or could not be reached; the message is safe to show (never the token)."""


class SlackClient:
    def __init__(self, token: str, transport: httpx.AsyncBaseTransport | None = None):
        self._token = token
        self._transport = transport

    async def _call(self, method: str, payload: dict, get: bool = False) -> dict:
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=TIMEOUT_S, headers=headers) as client:
                if get:
                    response = await client.get(f"{API}/{method}", params=payload)
                else:
                    response = await client.post(f"{API}/{method}", json=payload)
        except httpx.HTTPError as exc:
            raise SlackError(f"Cannot reach Slack: {type(exc).__name__}") from exc
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code >= 300 or not body.get("ok"):
            raise SlackError(f"Slack answered: {body.get('error') or response.status_code}")
        return body

    async def post(self, channel: str, text: str, blocks: list) -> dict:
        """Post a message; returns {"channel": "C…", "ts": "…"}. `channel` is an id or a name."""
        body = await self._call("chat.postMessage", {"channel": channel, "text": text, "blocks": blocks})
        return {"channel": body["channel"], "ts": body["ts"]}

    async def update(self, channel: str, ts: str, text: str, blocks: list) -> None:
        await self._call("chat.update", {"channel": channel, "ts": ts, "text": text, "blocks": blocks})

    async def permalink(self, channel: str, ts: str) -> str | None:
        try:
            body = await self._call("chat.getPermalink", {"channel": channel, "message_ts": ts}, get=True)
        except SlackError:
            return None  # the message is sent; only the link is missing
        return body.get("permalink")

    async def members(self) -> list[dict]:
        """Workspace members (people only): [{"id", "names": [real name, display name]}]."""
        body = await self._call("users.list", {"limit": 500}, get=True)
        people = []
        for m in body.get("members") or []:
            if m.get("deleted") or m.get("is_bot") or m.get("id") == "USLACKBOT":
                continue
            profile = m.get("profile") or {}
            names = [profile.get("real_name"), profile.get("display_name"), m.get("real_name")]
            people.append({"id": m["id"], "names": [n for n in names if n]})
        return people
