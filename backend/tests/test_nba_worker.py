"""One turn at a time per call: lines arriving during a turn give exactly one more turn."""
import asyncio

from app.nba.worker import NbaWorker


async def test_requests_during_a_turn_coalesce_into_one_more_turn():
    seen, latest, gate = [], {"n": 0}, asyncio.Event()

    async def run():
        seen.append(latest["n"])
        if len(seen) == 1:
            await gate.wait()

    worker = NbaWorker(run)
    latest["n"] = 1
    worker.submit()
    await asyncio.sleep(0)
    for n in (2, 3, 4):
        latest["n"] = n
        worker.submit()
    gate.set()
    await worker.wait()
    assert seen == [1, 4]


async def test_a_failing_turn_does_not_stop_later_turns():
    calls = []

    async def run():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("model down")

    worker = NbaWorker(run)
    worker.submit()
    await worker.wait()
    worker.submit()
    await worker.wait()
    assert len(calls) == 2


async def test_idle_worker_starts_right_away():
    ran = asyncio.Event()

    async def run():
        ran.set()

    NbaWorker(run).submit()
    await asyncio.wait_for(ran.wait(), 1)
