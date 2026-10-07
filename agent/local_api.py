"""Local HTTP endpoint (127.0.0.1 only) through which the VoiceBridge web app hands the agent its sign-in.

Only listed web-app origins are accepted, so other websites open in the browser cannot drive the agent.
Chrome asks before a page talks to a local address (Private Network Access); the preflight answers it.
"""
from typing import Awaitable, Callable

from aiohttp import web

LOCAL_PORT = 47800


def make_local_app(
    *,
    on_connect: Callable[[str, str], Awaitable[None]],
    status: Callable[[], dict],
    allowed_origins: list[str],
) -> web.Application:
    allowed = {o.rstrip("/") for o in allowed_origins}

    def cors(request: web.Request) -> dict | None:
        origin = request.headers.get("Origin", "")
        if origin not in allowed:
            return None
        return {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}

    @web.middleware
    async def origin_check(request: web.Request, handler):
        headers = cors(request)
        if headers is None:
            return web.json_response({"error": "origin not allowed"}, status=403)
        if request.method == "OPTIONS":
            return web.Response(
                status=204,
                headers=headers
                | {
                    "Access-Control-Allow-Methods": "GET, POST",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Access-Control-Allow-Private-Network": "true",
                },
            )
        response = await handler(request)
        response.headers.update(headers)
        return response

    async def connect(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            server, token = str(body["server"]), str(body["token"])
        except (ValueError, KeyError, TypeError):
            return web.json_response({"error": "expected {server, token}"}, status=400)
        await on_connect(server, token)
        return web.json_response(status())

    async def get_status(request: web.Request) -> web.Response:
        return web.json_response(status())

    async def preflight(request: web.Request) -> web.Response:  # answered by the middleware
        return web.Response(status=204)

    app = web.Application(middlewares=[origin_check])
    app.router.add_post("/connect", connect)
    app.router.add_get("/status", get_status)
    app.router.add_route("OPTIONS", "/{tail:.*}", preflight)
    return app
