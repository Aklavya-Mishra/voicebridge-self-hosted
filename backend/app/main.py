"""ASGI app factory: FastAPI (REST) wrapped by Socket.IO."""
import logging

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.config import DEV_JWT_SECRET, Settings
from app.llm import LLM, LlmChain, OllamaLLM, make_llm_chain
from app.nba.service import NbaService
from app.sessions import SessionStore
from app.socket_events import SocketHandlers, room_for
from app.stt import STTProvider, make_stt

logger = logging.getLogger("voicebridge")


def create_app(
    settings: Settings | None = None,
    stt: STTProvider | None = None,
    llm: LlmChain | None = None,
    nba_llm: LLM | None = None,
) -> socketio.ASGIApp:
    settings = settings or Settings.from_env()
    if settings.jwt_secret == DEV_JWT_SECRET:
        logger.warning("JWT_SECRET is the development default; set a real secret outside local dev")
    stt = stt or make_stt(settings)
    llm = llm or make_llm_chain(settings)
    if settings.call_analysis:
        logger.info("Call analysis with: %s", " > ".join(llm.names))
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
    api.state.llm = llm
    api.include_router(router)

    sio = socketio.AsyncServer(
        async_mode="asgi",
        cors_allowed_origins=list(settings.cors_origins),
        max_http_buffer_size=settings.max_chunk_bytes * 4,
    )

    nba = None
    if settings.nba_enabled:
        # Live hints use the local model only, whatever cloud keys are set.
        nba_llm = nba_llm or OllamaLLM(settings, model=settings.nba_model, num_ctx=settings.nba_num_ctx)
        logger.info("Live hints with: ollama %s", settings.nba_model)

        async def nba_emit(code: str, event: str, payload: dict) -> None:
            await sio.emit(event, payload, room=room_for(code))

        nba = NbaService(store, settings, nba_llm, nba_emit)
    SocketHandlers(sio, store, settings, stt, llm, nba)
    # At start, have the hint model read its fixed prompt (~20 s on CPU) before the first call needs it.
    return socketio.ASGIApp(sio, other_asgi_app=api, on_startup=nba.warm if nba else None)
