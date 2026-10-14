"""VoiceBridge desktop agent (Windows).

Install once and leave it running (tray icon; starts with Windows when run as the .exe). Open the
VoiceBridge web app on the same PC: the page hands the agent your sign-in over 127.0.0.1, and the agent
connects to VoiceBridge as you. When you click Start in a call, it streams your microphone (left channel,
you) and everything the PC plays (right channel, the other person) from any call app, with nothing shared.

Dev run: python voicebridge_agent.py [--origin http://localhost:5173]
"""
import argparse
import asyncio
import logging
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pyaudiowpatch as pa
import socketio
from aiohttp import web

from local_api import LOCAL_PORT, make_local_app
from mixer import StereoMixer

APP_NAME = "VoiceBridgeAgent"
DATA_DIR = Path(os.environ.get("APPDATA", Path.home())) / "VoiceBridge"
DEFAULT_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
SEND_INTERVAL_S = 0.05
LEVEL_LOG_INTERVAL_S = 10.0
# The agent tells the server every few seconds that it is here and which call it records. After a reconnect
# the server would otherwise not know either, and the other person's audio would be dropped.
HEARTBEAT_INTERVAL_S = 5.0
FROZEN = getattr(sys, "frozen", False)  # running as the PyInstaller .exe

logger = logging.getLogger("voicebridge-agent")


class Capture:
    """Default mic + loopback of every output device through WASAPI, feeding a StereoMixer from the audio threads.

    All outputs are captured because the call app may play on any of them (speakers, HDMI monitor, headset).
    """

    def __init__(self):
        self._audio = pa.PyAudio()
        wasapi = self._audio.get_host_api_info_by_type(pa.paWASAPI)
        self.mic = self._audio.get_device_info_by_index(wasapi["defaultInputDevice"])
        default_loop = self._audio.get_default_wasapi_loopback()
        others = [d for d in self._audio.get_loopback_device_info_generator() if d["index"] != default_loop["index"]]
        self.loops = [default_loop, *others]
        self.mixer = StereoMixer(
            mic_rate=int(self.mic["defaultSampleRate"]),
            mic_channels=self.mic["maxInputChannels"],
            loop_rate=int(default_loop["defaultSampleRate"]),
            loop_channels=default_loop["maxInputChannels"],
            extra_loops=[(int(d["defaultSampleRate"]), d["maxInputChannels"]) for d in others],
        )
        self.peaks = [0.0, 0.0]
        self._streams = []

    def _open(self, device: dict, add, slot: int):
        def callback(data, frames, time_info, status):
            samples = np.frombuffer(data, dtype=np.float32)
            if samples.size:
                self.peaks[slot] = max(self.peaks[slot], float(np.abs(samples).max()))
                add(samples)
            return None, pa.paContinue

        rate = int(device["defaultSampleRate"])
        return self._audio.open(
            format=pa.paFloat32,
            channels=device["maxInputChannels"],
            rate=rate,
            input=True,
            input_device_index=device["index"],
            frames_per_buffer=rate // 20,
            stream_callback=callback,
        )

    def start(self) -> None:
        self._streams = [self._open(self.mic, self.mixer.add_mic, 0)]
        for source, device in enumerate(self.loops):
            try:
                self._streams.append(self._open(device, lambda x, s=source: self.mixer.add_loop(x, s), 1))
            except OSError as exc:  # e.g. an HDMI output with no display attached
                logger.warning("Skipping output '%s': %s", device["name"], exc)
        logger.info("Capturing mic '%s' + outputs %s", self.mic["name"], [d["name"] for d in self.loops])

    def stop(self) -> None:
        for stream in self._streams:
            stream.stop_stream()
            stream.close()
        self._streams = []
        self._audio.terminate()


class Agent:
    def __init__(self, on_change=lambda text: None):
        self._sio: socketio.AsyncClient | None = None
        self._target: tuple[str, str] | None = None  # (server, token) currently used
        self._capture: Capture | None = None
        self._sender: asyncio.Task | None = None
        self._code: str | None = None
        self._heartbeat: asyncio.Task | None = None
        self._on_change = on_change

    def status(self) -> dict:
        return {"connected": bool(self._sio and self._sio.connected), "recording": self._code is not None}

    def _announce(self) -> None:
        s = self.status()
        self._on_change("Recording" if s["recording"] else "Connected" if s["connected"] else "Waiting for the web app")

    async def connect(self, server: str, token: str) -> None:
        """Called when the web app hands over its sign-in (every few seconds). Reconnects only if something changed.

        A dropped connection is not a change: the client reconnects by itself and keeps recording. Starting a
        second connection here stopped the recording, and the two connections then took turns pushing each
        other off the server, which dropped the other person's audio."""
        if self._target == (server, token) and self._sio:
            return
        self._target = (server, token)
        asyncio.create_task(self._reconnect(server, token))

    async def _reconnect(self, server: str, token: str) -> None:
        if self._code:
            await self._finish()
        if self._sio:
            await self._sio.disconnect()
        sio = socketio.AsyncClient(reconnection=True)

        async def on_connect() -> None:
            logger.info("Connected to %s", server)
            self._announce()
            await self._ping(sio)  # a reconnect: tell the server at once which call is being recorded

        sio.on("connect", on_connect)
        sio.on("disconnect", lambda *a: self._announce())
        sio.on("start_recording", self._start_recording)
        sio.on("stop_recording", self._stop_recording)
        self._sio = sio
        if self._heartbeat is None or self._heartbeat.done():
            self._heartbeat = asyncio.create_task(self._beat())
        try:
            await sio.connect(server, auth={"token": token, "client": "agent"}, transports=["websocket"])
        except socketio.exceptions.ConnectionError as exc:
            logger.warning("Cannot connect to %s: %s (open the web app again to retry)", server, exc)
            self._target = None
        self._announce()

    async def _start_recording(self, data: dict) -> dict:
        if self._code and self._code == data.get("code") and self._sender and not self._sender.done():
            return {"ok": True}  # asked again after a reconnect: keep recording
        if self._code:
            await self._finish()
        try:
            self._capture = Capture()
            self._capture.start()
        except Exception as exc:
            logger.exception("Could not open audio devices")
            self._capture = None
            return {"ok": False, "error": f"Agent could not open audio devices: {exc}"}
        self._code = data["code"]
        self._sender = asyncio.create_task(self._send(self._code))
        logger.info("Recording for session %s", self._code)
        self._announce()
        return {"ok": True}

    async def _stop_recording(self, data: dict) -> None:
        if data.get("code") == self._code:
            await self._finish()

    async def _finish(self) -> None:
        code, capture, sender, sio = self._code, self._capture, self._sender, self._sio
        self._code = self._capture = self._sender = None
        if sender:
            sender.cancel()
        if capture:
            capture.stop()
            if sio and sio.connected:
                for chunk in capture.mixer.chunks():
                    await sio.emit("audio_chunk", {"code": code, "audio": chunk, "channels": 2})
        if sio and sio.connected:
            await sio.emit("stop_audio", {"code": code})
        logger.info("Stopped recording for session %s", code)
        self._announce()

    async def _ping(self, sio: socketio.AsyncClient) -> None:
        try:
            await sio.emit("agent_ping", {"code": self._code})
        except Exception as exc:  # not connected right now; the next beat tries again
            logger.debug("Heartbeat not sent: %s", exc)

    async def _beat(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)
            sio = self._sio
            if sio and sio.connected:
                await self._ping(sio)

    async def _send(self, code: str) -> None:
        capture = self._capture
        since_log = 0.0
        while True:
            await asyncio.sleep(SEND_INTERVAL_S)
            chunks = capture.mixer.chunks()
            sio = self._sio
            # While the connection is down the audio is dropped; sending must not stop for good.
            if sio and sio.connected:
                for chunk in chunks:
                    try:
                        await sio.emit("audio_chunk", {"code": code, "audio": chunk, "channels": 2})
                    except Exception as exc:
                        logger.warning("Audio not sent: %s", exc)
                        break
            since_log += SEND_INTERVAL_S
            if since_log >= LEVEL_LOG_INTERVAL_S:
                mic, call = capture.peaks
                logger.info("Levels (last %.0fs): mic %.2f, call audio %.2f", since_log, mic, call)
                capture.peaks = [0.0, 0.0]
                since_log = 0.0


def enable_autostart() -> None:
    """Start the .exe with Windows (Task Manager > Startup apps can turn it off)."""
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, f'"{sys.executable}"')


class Tray:
    """What the user sees: tray icon state and Windows notifications. Plain logging when not the .exe."""

    def set_state(self, text: str) -> None:
        logger.info("State: %s", text)

    def notify(self, text: str) -> None:
        logger.info(text)

    def close(self) -> None:
        pass


def start_tray(stop: threading.Event) -> Tray:
    """System-tray icon with the agent's state and Quit. Runs on its own thread."""
    import pystray
    from PIL import Image, ImageDraw

    # Same mark as the web app's favicon: an indigo tile, an arch (the bridge) over sound-wave bars.
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, 63, 63), radius=16, fill=(79, 70, 229, 255))
    draw.arc((8, 16, 56, 64), start=205, end=335, fill=(199, 210, 254, 255), width=4)
    for x, top in ((17, 30), (24, 26), (30, 24), (36, 26), (43, 30)):
        draw.rounded_rectangle((x, top, x + 4, 49), radius=2, fill=(255, 255, 255, 255))
    icon = pystray.Icon(APP_NAME, image, "VoiceBridge Agent", menu=pystray.Menu(
        pystray.MenuItem(lambda item: f"VoiceBridge: {icon.title.split(': ', 1)[-1]}", None, enabled=False),
        pystray.MenuItem("Quit", lambda: (stop.set(), icon.stop())),
    ))
    icon.run_detached()

    class IconTray(Tray):
        def set_state(self, text: str) -> None:
            super().set_state(text)
            icon.title = f"VoiceBridge Agent: {text}"
            icon.update_menu()

        def notify(self, text: str) -> None:
            super().notify(text)
            icon.notify(text, "VoiceBridge Agent")

        def close(self) -> None:
            icon.stop()

    return IconTray()


async def serve(origins: list[str], stop: threading.Event, tray: Tray) -> None:
    agent = Agent(tray.set_state)
    runner = web.AppRunner(make_local_app(on_connect=agent.connect, status=agent.status, allowed_origins=origins))
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", LOCAL_PORT).start()
    except OSError:
        tray.notify("VoiceBridge Agent is already running (see the tray icon).")
        await asyncio.sleep(3)  # let the notification show before the icon goes away
        return
    logger.info("Agent ready on 127.0.0.1:%d; open the VoiceBridge web app (%s)", LOCAL_PORT, ", ".join(origins))
    tray.set_state("Waiting for the web app")
    tray.notify("VoiceBridge Agent is running in the tray. Open the VoiceBridge web app to connect.")
    while not stop.is_set():
        await asyncio.sleep(0.5)
    if agent.status()["recording"]:
        await agent._finish()
    await runner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceBridge desktop agent (Windows)")
    parser.add_argument("--origin", action="append", help=f"web app origin allowed to connect (default: {DEFAULT_ORIGINS})")
    args = parser.parse_args()
    origins = args.origin or os.environ.get("VOICEBRIDGE_WEB_ORIGINS", "").split(",")
    origins = [o.strip() for o in origins if o.strip()] or DEFAULT_ORIGINS

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(DATA_DIR / "agent.log", encoding="utf-8")]
    if not FROZEN:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", handlers=handlers)

    stop = threading.Event()
    tray = Tray()
    if FROZEN:
        enable_autostart()
        tray = start_tray(stop)
    try:
        asyncio.run(serve(origins, stop, tray))
    except KeyboardInterrupt:
        pass
    finally:
        tray.close()  # the tray thread would otherwise keep the process alive


if __name__ == "__main__":
    main()
