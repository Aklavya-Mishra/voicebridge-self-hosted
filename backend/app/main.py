"""ASGI app factory: FastAPI (REST) wrapped by Socket.IO."""
import logging

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.config import DEV_JWT_SECRET, Settings
from app.sessions import SessionStore
from app.socket_events import SocketHandlers
from app.stt import STTProvider, make_stt

logger = logging.getLogger("voicebridge")


def create_app(settings: Settings | None = None, stt: STTProvider | None = None) -> socketio.ASGIApp:
    settings = settings or Settings.from_env()
    if settings.jwt_secret == DEV_JWT_SECRET:
        logger.warning("JWT_SECRET is the development default; set a real secret outside local dev")
    stt = stt or make_stt(settings)
    store = SessionStore(max_participants=settings.max_participants, db_path=settings.db_path)

    api = FastAPI(title="VoiceBridge", version="0.1.0")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api.state.settings = settings
    api.state.stt = stt
    api.state.store = store
    api.include_router(router)

    sio = socketio.AsyncServer(
        async_mode="asgi",
        cors_allowed_origins=list(settings.cors_origins),
        max_http_buffer_size=settings.max_chunk_bytes * 4,
    )
    SocketHandlers(sio, store, settings, stt)
    return socketio.ASGIApp(sio, other_asgi_app=api)
