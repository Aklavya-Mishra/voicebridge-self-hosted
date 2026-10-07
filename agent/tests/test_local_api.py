"""The agent's local endpoint: only the VoiceBridge web app may hand it a sign-in."""
import pytest
from aiohttp.test_utils import TestClient, TestServer

from local_api import make_local_app

WEB = "http://localhost:5173"


@pytest.fixture
async def client():
    calls = []

    async def on_connect(server: str, token: str) -> None:
        calls.append((server, token))

    app = make_local_app(on_connect=on_connect, status=lambda: {"connected": bool(calls)}, allowed_origins=[WEB])
    async with TestClient(TestServer(app)) as c:
        c.calls = calls
        yield c


async def test_web_app_hands_over_sign_in(client):
    r = await client.post("/connect", json={"server": WEB, "token": "t1"}, headers={"Origin": WEB})
    assert r.status == 200
    assert r.headers["Access-Control-Allow-Origin"] == WEB
    assert client.calls == [(WEB, "t1")]
    status = await client.get("/status", headers={"Origin": WEB})
    assert (await status.json()) == {"connected": True}


async def test_other_websites_are_refused(client):
    r = await client.post("/connect", json={"server": "https://evil.example", "token": "x"},
                          headers={"Origin": "https://evil.example"})
    assert r.status == 403
    assert client.calls == []
    assert (await client.get("/status", headers={"Origin": "https://evil.example"})).status == 403


async def test_requests_without_origin_are_refused(client):
    assert (await client.post("/connect", json={"server": WEB, "token": "x"})).status == 403


async def test_preflight_allows_private_network_access(client):
    r = await client.options("/connect", headers={
        "Origin": WEB,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Private-Network": "true",
    })
    assert r.status == 204
    assert r.headers["Access-Control-Allow-Private-Network"] == "true"
    assert r.headers["Access-Control-Allow-Origin"] == WEB


async def test_bad_body_is_rejected(client):
    r = await client.post("/connect", json={"token": "x"}, headers={"Origin": WEB})
    assert r.status == 400
    assert client.calls == []
