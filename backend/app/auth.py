"""JWT issue/verify. v1 login is display-name only; the token carries user id and name."""
import time
import uuid

import jwt

from app.config import Settings

MAX_NAME_LENGTH = 40


class AuthError(Exception):
    pass


def issue_token(settings: Settings, name: str) -> tuple[str, dict]:
    name = (name or "").strip()
    if not name or len(name) > MAX_NAME_LENGTH:
        raise AuthError(f"Name must be 1-{MAX_NAME_LENGTH} characters")
    user = {"id": uuid.uuid4().hex, "name": name}
    now = int(time.time())
    claims = {"sub": user["id"], "name": user["name"], "iat": now, "exp": now + settings.jwt_ttl_seconds}
    token = jwt.encode(claims, settings.jwt_secret, algorithm="HS256")
    return token, user


def verify_token(settings: Settings, token: str | None) -> dict:
    if not token:
        raise AuthError("Missing token")
    try:
        claims = jwt.decode(
            token, settings.jwt_secret, algorithms=["HS256"], options={"require": ["sub", "exp"]}
        )
    except jwt.PyJWTError as exc:
        raise AuthError("Invalid token") from exc
    return {"id": claims["sub"], "name": claims.get("name", "")}
