"""Runs one call's next-best-action turns one at a time. Lines that arrive during a turn are not dropped and
do not pile up: they cause exactly one more turn, which reads the latest transcript."""
import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger("voicebridge")


class NbaWorker:
    def __init__(self, run: Callable[[], Awaitable[None]]):
        self._run = run
        self._task: asyncio.Task | None = None
        self._again = False

    def submit(self) -> None:
        if self._task is not None and not self._task.done():
            self._again = True
            return
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while True:
            self._again = False
            try:
                await self._run()
            except Exception:
                logger.exception("Next-best-action turn failed")
            if not self._again:
                return

    async def wait(self) -> None:
        """Until the current turn (and the one queued after it) is done."""
        if self._task is not None:
            await self._task
