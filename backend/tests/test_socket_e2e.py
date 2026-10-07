"""Two Socket.IO clients against a real uvicorn server: the mirrored-chat round trip."""
import asyncio
import socket

import httpx
import pytest
import socketio
import uvicorn

from app.main import create_app
from tests.conftest import RecordingSTT, chunks, silence, tone


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def server(settings):
    port = free_port()
    app = create_app(settings, stt=RecordingSTT(text="hello"))
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(srv.serve())
    while not srv.started:
        await asyncio.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    await task


async def login(base: str, name: str) -> tuple[str, dict]:
    async with httpx.AsyncClient(base_url=base) as c:
        body = (await c.post("/api/auth/login", json={"name": name})).json()
        return body["token"], body["user"]


async def connect(base: str, token: str):
    client = socketio.AsyncClient()
    events: list[tuple[str, dict]] = []
    client.on("transcript", lambda d: events.append(("transcript", d)))
    client.on("participants", lambda d: events.append(("participants", d)))
    client.on("agent_status", lambda d: events.append(("agent_status", d)))
    await client.connect(base, auth={"token": token}, transports=["websocket"])
    return client, events


async def test_unauthenticated_socket_refused(server):
    client = socketio.AsyncClient()
    with pytest.raises(socketio.exceptions.ConnectionError):
        await client.connect(server, auth={"token": "junk"}, transports=["websocket"])


async def test_two_users_see_each_others_speech(server):
    token_a, user_a = await login(server, "Asha")
    token_b, user_b = await login(server, "Ben")
    async with httpx.AsyncClient(base_url=server) as c:
        code = (await c.post("/api/sessions", headers={"Authorization": f"Bearer {token_a}"})).json()["session"]["code"]

    a, events_a = await connect(server, token_a)
    b, events_b = await connect(server, token_b)
    assert (await a.call("join_session", {"code": code}))["ok"]
    joined = await b.call("join_session", {"code": code})
    assert joined["ok"] and len(joined["session"]["participants"]) == 2

    # Asha speaks, then Ben answers.
    for chunk in chunks(tone(1.0) + silence(1.0)):
        await a.emit("audio_chunk", {"code": code, "audio": chunk})
    await asyncio.sleep(0.3)
    for chunk in chunks(tone(1.0) + silence(1.0)):
        await b.emit("audio_chunk", {"code": code, "audio": chunk})
    await a.emit("stop_audio", {"code": code})
    await b.emit("stop_audio", {"code": code})

    for _ in range(50):
        finals_a = [d for kind, d in events_a if kind == "transcript" and d["final"] and d["text"]]
        finals_b = [d for kind, d in events_b if kind == "transcript" and d["final"] and d["text"]]
        if len(finals_a) == 2 and len(finals_b) == 2:
            break
        await asyncio.sleep(0.1)

    # Both sides receive both utterances, attributed to the right speaker, in speaking order.
    for finals in (finals_a, finals_b):
        ordered = sorted(finals, key=lambda d: d["start_ts"])
        assert [d["speaker"]["id"] for d in ordered] == [user_a["id"], user_b["id"]]

    async with httpx.AsyncClient(base_url=server) as c:
        r = await c.get(f"/api/sessions/{code}", headers={"Authorization": f"Bearer {token_b}"})
    assert [u["speaker_id"] for u in r.json()["session"]["transcript"]] == [user_a["id"], user_b["id"]]

    # A third user cannot join.
    token_c, _ = await login(server, "Cara")
    c_client, _ = await connect(server, token_c)
    assert (await c_client.call("join_session", {"code": code}))["ok"] is False
    for client in (a, b, c_client):
        await client.disconnect()


def stereo(left: bytes, right: bytes) -> bytes:
    import numpy as np

    return np.stack([np.frombuffer(left, np.int16), np.frombuffer(right, np.int16)], axis=1).tobytes()


async def test_stereo_mic_plus_call_audio_splits_speakers_by_channel(server):
    """One user streams mic (left) + call audio (right): the call side becomes 'Other person'."""
    token, user = await login(server, "Aklavya")
    async with httpx.AsyncClient(base_url=server) as c:
        code = (await c.post("/api/sessions", headers={"Authorization": f"Bearer {token}"})).json()["session"]["code"]
    client, events = await connect(server, token)
    assert (await client.call("join_session", {"code": code}))["ok"]

    # Aklavya speaks first on the mic, then the remote person answers on the call channel.
    left = tone(1.0) + silence(2.0)
    right = silence(1.5) + tone(1.0) + silence(0.5)
    for chunk in chunks(stereo(left, right), chunk_s=0.05):  # 0.05 s of stereo = 0.1 s per channel's bytes
        await client.emit("audio_chunk", {"code": code, "audio": chunk, "channels": 2})
    await client.emit("stop_audio", {"code": code})

    for _ in range(50):
        finals = [d for kind, d in events if kind == "transcript" and d["final"] and d["text"]]
        if len(finals) == 2:
            break
        await asyncio.sleep(0.1)

    ordered = sorted(finals, key=lambda d: d["start_ts"])
    assert [d["speaker"]["id"] for d in ordered] == [user["id"], f"{user['id']}-call"]
    assert ordered[1]["speaker"]["name"] == "Other person"
    # Both channels share one clock: the call side starts ~1.5 s after the mic side.
    assert ordered[1]["start_ts"] - ordered[0]["start_ts"] == pytest.approx(1.5, abs=0.4)
    await client.disconnect()

