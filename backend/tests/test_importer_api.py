import io
import wave

import httpx
import numpy as np
import pytest

from app.main import create_app
from tests.conftest import SR, RecordingSTT, silence, tone


def wav_bytes(channels: list[bytes]) -> bytes:
    arrays = [np.frombuffer(c, dtype=np.int16) for c in channels]
    length = max(a.size for a in arrays)
    arrays = [np.pad(a, (0, length - a.size)) for a in arrays]
    interleaved = np.stack(arrays, axis=1).reshape(-1) if len(arrays) > 1 else arrays[0]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(len(arrays))
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(interleaved.astype(np.int16).tobytes())
    return buffer.getvalue()


# Left speaks at 0.5s and 4.0s, right speaks at 2.0s: a back-and-forth conversation.
LEFT = silence(0.5) + tone(1.0) + silence(2.5) + tone(1.0) + silence(1.0)
RIGHT = silence(2.0) + tone(1.0, freq=330) + silence(3.0)


@pytest.fixture
async def client(settings):
    app = create_app(settings, stt=RecordingSTT(text="words"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def login(client, name="Asha") -> dict:
    r = await client.post("/api/auth/login", json={"name": name})
    assert r.status_code == 200
    body = r.json()
    return {"headers": {"Authorization": f"Bearer {body['token']}"}, "user": body["user"]}


async def test_health(client):
    r = await client.get("/api/health")
    assert r.json() == {"status": "ok", "stt_provider": "fake", "nba": False}


async def test_session_endpoints_require_auth(client):
    assert (await client.post("/api/sessions")).status_code == 401
    assert (await client.post("/api/sessions", headers={"Authorization": "Bearer junk"})).status_code == 401


async def test_create_and_get_session(client):
    me = await login(client)
    session = (await client.post("/api/sessions", headers=me["headers"])).json()["session"]
    r = await client.get(f"/api/sessions/{session['code']}", headers=me["headers"])
    assert r.status_code == 200 and r.json()["session"]["participants"] == [me["user"]]
    other = await login(client, "Ben")
    assert (await client.get(f"/api/sessions/{session['code']}", headers=other["headers"])).status_code == 403


@pytest.mark.parametrize("me_channel,me_first", [("left", True), ("right", False)])
async def test_import_stereo_assigns_speakers_by_channel(client, me_channel, me_first):
    me = await login(client)
    r = await client.post(
        "/api/import",
        headers=me["headers"],
        files={"file": ("call.wav", wav_bytes([LEFT, RIGHT]), "audio/wav")},
        data={"me_channel": me_channel},
    )
    assert r.status_code == 200, r.text
    session = r.json()["session"]
    speakers = [u["speaker_id"] for u in session["transcript"]]
    me_id = me["user"]["id"]
    expected = [me_id, "other", me_id] if me_first else ["other", me_id, "other"]
    assert speakers == expected
    assert {p["id"] for p in session["participants"]} == {me_id, "other"}
    assert session["kind"] == "import"


async def test_import_rejects_mono(client):
    me = await login(client)
    r = await client.post(
        "/api/import",
        headers=me["headers"],
        files={"file": ("mono.wav", wav_bytes([LEFT]), "audio/wav")},
        data={"me_channel": "left"},
    )
    assert r.status_code == 422 and "stereo" in r.json()["detail"]


async def test_import_rejects_identical_channels(client):
    me = await login(client)
    r = await client.post(
        "/api/import",
        headers=me["headers"],
        files={"file": ("fake.wav", wav_bytes([LEFT, LEFT]), "audio/wav")},
        data={"me_channel": "left"},
    )
    assert r.status_code == 422 and "same audio" in r.json()["detail"]


async def test_import_rejects_garbage_and_bad_channel(client):
    me = await login(client)
    r = await client.post(
        "/api/import",
        headers=me["headers"],
        files={"file": ("x.wav", b"not audio", "audio/wav")},
        data={"me_channel": "left"},
    )
    assert r.status_code == 422
    r = await client.post(
        "/api/import",
        headers=me["headers"],
        files={"file": ("call.wav", wav_bytes([LEFT, RIGHT]), "audio/wav")},
        data={"me_channel": "middle"},
    )
    assert r.status_code == 422
