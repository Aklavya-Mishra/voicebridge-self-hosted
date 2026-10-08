"""Desktop agent: the start/stop handshake (web UI -> server -> agent), over a real server.

The agent connects with the web UI's own token (the web app hands it over on the local machine)."""
import asyncio

import httpx
import socketio

from tests.conftest import chunks, silence, tone
from tests.test_socket_e2e import connect, login, server, stereo  # noqa: F401  (server is a fixture)


class FakeAgent:
    """Stands in for the desktop agent: answers start_recording and records stop_recording."""

    def __init__(self, ok: bool = True):
        self.client = socketio.AsyncClient()
        self.started: list[str] = []
        self.stopped: list[str] = []
        self._ok = ok
        self.client.on("start_recording", self._start)
        self.client.on("stop_recording", lambda d: self.stopped.append(d["code"]))

    async def _start(self, data):
        self.started.append(data["code"])
        return {"ok": True} if self._ok else {"ok": False, "error": "No microphone"}

    async def connect(self, base: str, token: str):
        await self.client.connect(base, auth={"token": token, "client": "agent"}, transports=["websocket"])


async def new_session(base: str, token: str) -> str:
    async with httpx.AsyncClient(base_url=base) as c:
        r = await c.post("/api/sessions", headers={"Authorization": f"Bearer {token}"})
        return r.json()["session"]["code"]


def statuses(events) -> list[dict]:
    return [d for kind, d in events if kind == "agent_status"]


async def wait_for(predicate, tries: int = 50):
    for _ in range(tries):
        if predicate():
            return True
        await asyncio.sleep(0.1)
    return False


async def test_web_starts_agent_which_streams_both_sides(server):
    web_token, user = await login(server, "Aklavya")
    agent_token = web_token
    code = await new_session(server, web_token)

    web, events = await connect(server, web_token)
    assert (await web.call("join_session", {"code": code}))["ok"]

    agent = FakeAgent()
    await agent.connect(server, agent_token)
    assert await wait_for(lambda: statuses(events) and statuses(events)[-1]["online"])

    ack = await web.call("agent_start", {"code": code})
    assert ack == {"ok": True}
    assert agent.started == [code]

    # The agent streams mic (left) + loopback (right) into the session it was started for.
    for chunk in chunks(stereo(tone(1.0) + silence(2.0), silence(1.5) + tone(1.0) + silence(0.5)), chunk_s=0.05):
        await agent.client.emit("audio_chunk", {"code": code, "audio": chunk, "channels": 2})

    assert (await web.call("agent_stop", {"code": code})) == {"ok": True}
    assert await wait_for(lambda: agent.stopped == [code])
    await agent.client.emit("stop_audio", {"code": code})

    def finals():
        return [d for kind, d in events if kind == "transcript" and d["final"] and d["text"]]

    assert await wait_for(lambda: len(finals()) == 2)
    assert [d["speaker"]["id"] for d in sorted(finals(), key=lambda d: d["start_ts"])] == [user["id"], f"{user['id']}-call"]

    await agent.client.disconnect()
    assert await wait_for(lambda: not statuses(events)[-1]["online"])
    await web.disconnect()


async def test_agent_start_without_agent_running(server):
    web_token, _ = await login(server, "Aklavya")
    code = await new_session(server, web_token)
    web, events = await connect(server, web_token)
    assert (await web.call("join_session", {"code": code}))["ok"]
    assert await wait_for(lambda: statuses(events) == [{"online": False}])
    ack = await web.call("agent_start", {"code": code})
    assert ack["ok"] is False and "agent" in ack["error"].lower()
    await web.disconnect()


async def test_agent_start_reports_agent_failure(server):
    web_token, _ = await login(server, "Aklavya")
    agent_token = web_token
    code = await new_session(server, web_token)
    web, _ = await connect(server, web_token)
    assert (await web.call("join_session", {"code": code}))["ok"]
    agent = FakeAgent(ok=False)
    await agent.connect(server, agent_token)
    assert (await web.call("agent_start", {"code": code})) == {"ok": False, "error": "No microphone"}
    await agent.client.disconnect()
    await web.disconnect()


async def test_agent_cannot_stream_into_a_session_it_was_not_started_for(server):
    web_token, _ = await login(server, "Aklavya")
    agent_token = web_token
    code = await new_session(server, web_token)
    web, events = await connect(server, web_token)
    assert (await web.call("join_session", {"code": code}))["ok"]
    agent = FakeAgent()
    await agent.connect(server, agent_token)
    for chunk in chunks(stereo(tone(1.0) + silence(1.0), silence(2.0)), chunk_s=0.05):
        await agent.client.emit("audio_chunk", {"code": code, "audio": chunk, "channels": 2})
    await agent.client.emit("stop_audio", {"code": code})
    await asyncio.sleep(0.5)
    assert not [d for kind, d in events if kind == "transcript"]
    await agent.client.disconnect()
    await web.disconnect()


async def test_leaving_the_call_stops_the_agent(server):
    web_token, _ = await login(server, "Aklavya")
    agent_token = web_token
    code = await new_session(server, web_token)
    web, _ = await connect(server, web_token)
    assert (await web.call("join_session", {"code": code}))["ok"]
    agent = FakeAgent()
    await agent.connect(server, agent_token)
    assert (await web.call("agent_start", {"code": code}))["ok"]
    await web.emit("leave_session", {"code": code})
    assert await wait_for(lambda: agent.stopped == [code])
    await agent.client.disconnect()
    await web.disconnect()


async def test_muting_my_mic_keeps_transcribing_the_other_person(server):
    """Stop mic with the agent mutes only my side: the call audio is still transcribed."""
    web_token, user = await login(server, "Aklavya")
    code = await new_session(server, web_token)
    web, events = await connect(server, web_token)
    assert (await web.call("join_session", {"code": code}))["ok"]
    agent = FakeAgent()
    await agent.connect(server, web_token)
    assert await wait_for(lambda: statuses(events) and statuses(events)[-1]["online"])
    assert (await web.call("agent_start", {"code": code})) == {"ok": True}

    assert (await web.call("mute_mic", {"code": code, "muted": True})) == {"ok": True}
    # Both sides speak; only the call side may be transcribed.
    for chunk in chunks(stereo(tone(1.0) + silence(3.5), silence(1.5) + tone(1.0) + silence(2.0)), chunk_s=0.05):
        await agent.client.emit("audio_chunk", {"code": code, "audio": chunk, "channels": 2})

    def finals():
        return [d for kind, d in events if kind == "transcript" and d["final"] and d["text"]]

    assert await wait_for(lambda: len(finals()) >= 1)
    await asyncio.sleep(0.5)
    assert [d["speaker"]["id"] for d in finals()] == [f"{user['id']}-call"]
    assert agent.stopped == []  # the agent keeps recording

    # Unmuted again: my side is transcribed too.
    assert (await web.call("mute_mic", {"code": code, "muted": False})) == {"ok": True}
    for chunk in chunks(stereo(tone(1.0) + silence(1.0), silence(2.0)), chunk_s=0.05):
        await agent.client.emit("audio_chunk", {"code": code, "audio": chunk, "channels": 2})
    assert await wait_for(lambda: user["id"] in [d["speaker"]["id"] for d in finals()])

    assert (await web.call("mute_mic", {"code": "ZZZZZZ", "muted": True}))["ok"] is False
    await agent.client.disconnect()
    await web.disconnect()


async def test_muting_mid_sentence_saves_what_i_already_said(server):
    web_token, user = await login(server, "Aklavya")
    code = await new_session(server, web_token)
    web, events = await connect(server, web_token)
    assert (await web.call("join_session", {"code": code}))["ok"]
    agent = FakeAgent()
    await agent.connect(server, web_token)
    assert await wait_for(lambda: statuses(events) and statuses(events)[-1]["online"])
    assert (await web.call("agent_start", {"code": code})) == {"ok": True}
    for chunk in chunks(stereo(tone(1.0), silence(1.0)), chunk_s=0.05):  # no pause yet: sentence still open
        await agent.client.emit("audio_chunk", {"code": code, "audio": chunk, "channels": 2})
    await asyncio.sleep(0.2)
    assert (await web.call("mute_mic", {"code": code, "muted": True})) == {"ok": True}
    assert await wait_for(
        lambda: any(kind == "transcript" and d["final"] and d["text"] and d["speaker"]["id"] == user["id"] for kind, d in events)
    )
    await agent.client.disconnect()
    await web.disconnect()
