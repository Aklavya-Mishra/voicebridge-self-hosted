"""Live next-best-action hints for a call: instant rule hints, then one local-model turn per customer line.

SocketHandlers calls on_final() for every finished line. The call type (domain) is never picked by hand: it
follows the customer's latest line (app.nba.engine). Hints, facts and the domain go out through `emit(code, event, payload)` and are saved on the session.
Only the call-audio side (the other person, "Computer" until renamed) counts as the customer.
"""
import asyncio
import logging
from collections.abc import Awaitable, Callable

import httpx

from app.config import Settings
from app.nba.engine import SYSTEM, Line, NbaState, run_turn
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
        self._warming: asyncio.Task | None = None

    def warm(self) -> None:
        """Have the local model read the fixed hint prompt now, in the background, so the first hint of a call
        does not wait for it (~20 s on CPU). Called when a call is joined and after each call summary: Ollama
        drops the cached prompt when it unloads the model or runs another prompt (the summary)."""
        warm = getattr(self._llm, "warm", None)
        if warm is None or (self._warming is not None and not self._warming.done()):
            return

        async def run() -> None:
            try:
                await warm(SYSTEM)
            except Exception as exc:  # Ollama not running: the first turn reports it
                logger.info("Could not warm the live hint model: %s", exc)

        self._warming = asyncio.create_task(run())

    async def on_final(self, code: str, utterance: Utterance) -> None:
        if not is_call_speaker(utterance.speaker_id) or not utterance.text.strip():
            return
        state = self._state(code)
        # The call type can change with any line, so every pack's phrases apply (rule ids are unique across
        # packs); rule hints also work when the model is down.
        fired = state.fired()
        hints = []
        for pack in PACKS.values():
            hints += match_phrase_rules(pack, utterance.text, "customer", fired, self._settings.nba_phrase_match)
        await self._add_rule_hints(code, state, hints)
        self._worker(code).submit()

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
                detected=session.nba.get("detected"),
                facts=session.nba["facts"],
                fact_log=session.nba.setdefault("fact_log", {}),
                hints=session.nba["hints"],
                domain_since=session.nba.get("domain_since", 0),
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
        window = session.transcript[-self._settings.nba_context_lines:]
        # The advisor's talk before the caller first speaks in this window is not about the caller: keep only
        # the line just before (the question being answered), or the model takes the advisor's topic.
        first = next((i for i, u in enumerate(window) if is_call_speaker(u.speaker_id)), 0)
        lines = [
            Line("customer" if is_call_speaker(u.speaker_id) else "advisor", u.speaker_name, u.text)
            for u in window[max(first - 1, 0):]
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
            else:
                logger.warning("Live hint model error for session %s: %s", code, exc)
                await self._status(code, "unavailable", f"local model error (HTTP {exc.response.status_code})")
            return
        except TimeoutError:
            await self._status(code, "unavailable", f"model too slow (over {self._settings.nba_timeout_s:g} s)")
            return
        except httpx.TransportError as exc:  # Ollama stopped mid-request
            logger.warning("Live hint model unreachable for session %s: %s", code, exc)
            await self._status(code, "unavailable", "local model stopped responding")
            return
        except Exception:
            logger.exception("Next-best-action turn failed for session %s", code)
            await self._status(code, "idle")
            return

        customer_lines = self._customer_lines(code)
        if result.domain_changed:
            state.domain_since = customer_lines - 1  # the line that showed the new domain counts towards it
            await self._emit_domain(code, state)
        if result.facts_changed:
            await self._emit(code, "nba_facts", {"code": code, "facts": dict(state.facts)})
        if result.hint:
            await self._emit(code, "nba_hint", {"code": code, **result.hint})
        pack = PACKS.get(state.detected or "")
        if pack:
            hints = match_fact_rules(
                pack, state.facts, customer_lines - state.domain_since, state.fired(), self._settings.nba_absent_after
            )
            await self._add_rule_hints(code, state, hints)
        self._save(code, state)
        await self._status(code, "idle")

    def _customer_lines(self, code: str) -> int:
        return sum(1 for u in self._store.get(code).transcript if is_call_speaker(u.speaker_id))

    async def _add_rule_hints(self, code: str, state: NbaState, hints: list[dict]) -> None:
        for hint in hints:
            state.hints.append(hint)
            await self._emit(code, "nba_hint", {"code": code, **hint})
        if hints:
            self._save(code, state)

    def _save(self, code: str, state: NbaState) -> None:
        self._store.set_nba(
            code,
            {
                "hints": state.hints, "facts": state.facts, "detected": state.detected, "fact_log": state.fact_log,
                "domain_since": state.domain_since,
            },
        )

    async def _emit_domain(self, code: str, state: NbaState) -> None:
        await self._emit(code, "nba_domain", {"code": code, "domain": state.detected})

    async def _status(self, code: str, value: str, reason: str | None = None) -> None:
        payload = {"code": code, "state": value}
        if reason:
            payload["reason"] = reason
        await self._emit(code, "nba_status", payload)
