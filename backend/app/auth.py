"""JWT issue/verify. v1 login is display-name only; the token carries user id and name."""
import hashlib
import time

import jwt

from app.config import Settings

MAX_NAME_LENGTH = 40


class AuthError(Exception):
    pass


def issue_token(settings: Settings, name: str) -> tuple[str, dict]:
    name = (name or "").strip()
    if not name or len(name) > MAX_NAME_LENGTH:
        raise AuthError(f"Name must be 1-{MAX_NAME_LENGTH} characters")
    # Name-only sign-in: the same name is always the same user, so saved calls stay theirs after the
    # token expires. Replace with real accounts before more than one person shares a deployment.
    user = {"id": user_id_for(name), "name": name}
    now = int(time.time())
    claims = {"sub": user["id"], "name": user["name"], "iat": now, "exp": now + settings.jwt_ttl_seconds}
    token = jwt.encode(claims, settings.jwt_secret, algorithm="HS256")
    return token, user


def user_id_for(name: str) -> str:
    canonical = " ".join(name.split()).casefold()
    return hashlib.sha256(f"voicebridge-user:{canonical}".encode()).hexdigest()[:32]


def verify_token(settings: Settings, token: str | None) -> dict:
    if not token:
        raise AuthError("Missing token")
    try:
        claims = jwt.decode(
            token, settings.jwt_secret, algorithms=["HS256"], options={"require": ["sub", "exp"]}
        )
    except jwt.PyJWTError as exc:
        raise AuthError("Invalid token") from exc
    name = claims.get("name", "")
    # Tokens issued before ids were derived from the name carry a random id: map them to the stable one.
    return {"id": user_id_for(name) if name else claims["sub"], "name": name}
