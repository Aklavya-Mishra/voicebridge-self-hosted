"""REST endpoints: login, sessions, pre-recorded import."""
import asyncio
import time

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.analysis import analyse_session, mask_text
from app.auth import AuthError, issue_token, verify_token
from app.importer import OTHER_SPEAKER_ID, OTHER_SPEAKER_NAME, ImportRejected, transcribe_stereo
from app.sessions import SessionError

router = APIRouter(prefix="/api")


class LoginRequest(BaseModel):
    name: str


def current_user(request: Request, authorization: str | None = Header(default=None)) -> dict:
    token = authorization.removeprefix("Bearer ").strip() if authorization else None
    try:
        return verify_token(request.app.state.settings, token)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _session_error(exc: SessionError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


@router.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "stt_provider": request.app.state.stt.name}


@router.post("/auth/login")
def login(body: LoginRequest, request: Request) -> dict:
    try:
        token, user = issue_token(request.app.state.settings, body.name)
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"token": token, "user": user}


@router.get("/auth/me")
def me(user: dict = Depends(current_user)) -> dict:
    """Who the server thinks the token belongs to; the web app keeps its saved identity in sync with it."""
    return {"user": user}


@router.get("/sessions")
def list_sessions(request: Request, user: dict = Depends(current_user)) -> dict:
    """The signed-in user's saved calls, newest first."""
    return {"sessions": request.app.state.store.list_for(user["id"])}


@router.post("/sessions")
def create_session(request: Request, user: dict = Depends(current_user)) -> dict:
    return {"session": request.app.state.store.create(user).to_dict()}


@router.get("/sessions/{code}")
def get_session(code: str, request: Request, user: dict = Depends(current_user)) -> dict:
    try:
        return {"session": request.app.state.store.require_participant(code, user).to_dict()}
    except SessionError as exc:
        raise _session_error(exc) from exc


@router.post("/import")
async def import_recording(
    request: Request,
    file: UploadFile = File(...),
    me_channel: str = Form(...),
    user: dict = Depends(current_user),
) -> dict:
    settings = request.app.state.settings
    data = await file.read(settings.max_import_bytes + 1)
    if len(data) > settings.max_import_bytes:
        raise HTTPException(status_code=413, detail="File is too large")
    try:
        utterances = await transcribe_stereo(
            data, me_channel, user, settings, request.app.state.stt, started_at=time.time()
        )
    except ImportRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    store = request.app.state.store
    session = store.create(user, kind="import")
    store.add_guest(session.code, OTHER_SPEAKER_ID, OTHER_SPEAKER_NAME)
    for utterance in utterances:
        if settings.pii_masking:
            utterance.text = mask_text(utterance.text)
        store.add_utterance(session.code, utterance)
    if settings.call_analysis:
        asyncio.create_task(analyse_session(store, session.code, request.app.state.llm))
    return {"session": session.to_dict()}
