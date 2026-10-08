"""Live next-best-action hints for a call: instant rule hints, then one local-model turn per customer line.

SocketHandlers calls on_final() for every finished line and set_domain() for the web UI's call-type picker.
Hints, facts and the domain go out through `emit(code, event, payload)` and are saved on the session.
Only the call-audio side (the other person, "Computer" until renamed) counts as the customer.
"""
import logging
from collections.abc import Awaitable, Callable

import httpx

from app.config import Settings
from app.nba.engine import Line, NbaState, run_turn
from app.nba.knowledge import PACKS
from app.nba.rules import match_fact_rules, match_phrase_rules
from app.nba.worker import NbaWorker
from app.sessions import SessionStore, Utterance, is_call_speaker

logger = logging.getLogger("voicebridge")

Emit = Callable[[str, str, dict], Awaitable[None]]


class NbaService:
    def __init__(self, store: SessionStore, settings: Settings, llm, emit: Emit):
        self._store = store
        self._settings = settings
        self._llm = llm
        self._emit = emit
        self._states: dict[str, NbaState] = {}
        self._workers: dict[str, NbaWorker] = {}

    async def on_final(self, code: str, utterance: Utterance) -> None:
        if not is_call_speaker(utterance.speaker_id) or not utterance.text.strip():
            return
        state = self._state(code)
        pack = PACKS.get(state.active_domain or "")
        if pack:
            fired = state.fired()
            await self._add_rule_hints(
                code, state, match_phrase_rules(pack, utterance.text, "customer", fired, self._settings.nba_phrase_match)
            )
        self._worker(code).submit()

    async def set_domain(self, code: str, domain: str) -> None:
        if domain != "auto" and domain not in PACKS:
            raise ValueError("Unknown call type")
        state = self._state(code)
        state.domain, state.detected = domain, None
        self._store.set_domain(code, domain)
        self._save(code, state)
        await self._emit_domain(code, state)

    async def drain(self, code: str) -> None:
        """Until the call's running turn (and any queued one) is done."""
        worker = self._workers.get(code)
        if worker is not None:
            await worker.wait()

    def _state(self, code: str) -> NbaState:
        state = self._states.get(code)
        if state is None:
            session = self._store.get(code)
            # Share the session's hint list and facts dict, so a speaker rename is seen here.
            state = NbaState(
                domain=session.domain,
                detected=session.nba.get("detected"),
                facts=session.nba["facts"],
                hints=session.nba["hints"],
            )
            self._states[code] = state
        return state

    def _worker(self, code: str) -> NbaWorker:
        worker = self._workers.get(code)
        if worker is None:
            worker = self._workers[code] = NbaWorker(lambda: self._turn(code))
        return worker

    async def _turn(self, code: str) -> None:
        state = self._state(code)
        session = self._store.get(code)
        lines = [
            Line("customer" if is_call_speaker(u.speaker_id) else "advisor", u.speaker_name, u.text)
            for u in session.transcript[-self._settings.nba_context_lines:]
        ]
        await self._status(code, "thinking")
        try:
            result = await run_turn(state, lines, self._llm, self._settings)
        except httpx.ConnectError:
            await self._status(code, "unavailable", "local model not running")
            return
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                await self._status(code, "unavailable", f"model not installed: ollama pull {self._settings.nba_model}")
                return
            logger.exception("Next-best-action turn failed for session %s", code)
            await self._status(code, "idle")
            return
        except Exception:
            logger.exception("Next-best-action turn failed for session %s", code)
            await self._status(code, "idle")
            return

        if result.domain_changed:
            await self._emit_domain(code, state)
        if result.facts_changed:
            await self._emit(code, "nba_facts", {"code": code, "facts": dict(state.facts)})
        if result.hint:
            await self._emit(code, "nba_hint", {"code": code, **result.hint})
        pack = PACKS.get(state.active_domain or "")
        if pack:
            customer_lines = sum(1 for u in session.transcript if is_call_speaker(u.speaker_id))
            hints = match_fact_rules(pack, state.facts, customer_lines, state.fired(), self._settings.nba_absent_after)
            await self._add_rule_hints(code, state, hints)
        self._save(code, state)
        await self._status(code, "idle")

    async def _add_rule_hints(self, code: str, state: NbaState, hints: list[dict]) -> None:
        for hint in hints:
            state.hints.append(hint)
            await self._emit(code, "nba_hint", {"code": code, **hint})
        if hints:
            self._save(code, state)

    def _save(self, code: str, state: NbaState) -> None:
        self._store.set_nba(code, {"hints": state.hints, "facts": state.facts, "detected": state.detected})

    async def _emit_domain(self, code: str, state: NbaState) -> None:
        await self._emit(code, "nba_domain", {"code": code, "domain": state.active_domain, "auto": state.domain == "auto"})

    async def _status(self, code: str, value: str, reason: str | None = None) -> None:
        payload = {"code": code, "state": value}
        if reason:
            payload["reason"] = reason
        await self._emit(code, "nba_status", payload)
