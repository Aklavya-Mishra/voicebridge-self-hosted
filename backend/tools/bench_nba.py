"""Time live-hint turns on the local model, on this PC's CPU.

    .venv/Scripts/python tools/bench_nba.py qwen2.5:3b qwen2.5:7b

For each model and sample call: one warm-up turn (loads the model), then one turn per customer line (the call
type follows each line), printing seconds per turn, the detected domain, facts and hint.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.llm import OllamaLLM  # noqa: E402
from app.nba.engine import Line, NbaState, run_turn  # noqa: E402

CALLS = {
    "telecom": [
        ("advisor", "Aklavya", "Thank you for calling, how can I help you today?"),
        ("customer", "Computer", "Hi, my home internet has not been working since yesterday evening."),
        ("advisor", "Aklavya", "I'm sorry about that. Which area are you in?"),
        ("customer", "Computer", "Baner in Pune. It's the third time this month, I'm thinking of switching provider."),
        ("advisor", "Aklavya", "I understand. Can you tell me which plan you have?"),
        ("customer", "Computer", "The fibre 100 plan, I pay every month."),
    ],
    "sales": [
        ("advisor", "Aklavya", "Hi, this is Aklavya from the CRM team, is this a good time?"),
        ("customer", "Computer", "Yes, we are a 20 person agency and our leads are all in spreadsheets."),
        ("advisor", "Aklavya", "Got it. What made you look at a CRM now?"),
        ("customer", "Computer", "We lost two clients last month because nobody followed up."),
        ("advisor", "Aklavya", "Our team plan would cover all 20 people."),
        ("customer", "Computer", "Honestly that sounds too expensive for us, another tool is half the price."),
    ],
}


async def bench(model: str) -> None:
    settings = Settings(nba_model=model, nba_timeout_s=600)
    llm = OllamaLLM(settings, model=model, num_ctx=settings.nba_num_ctx)
    print(f"\n=== {model}")
    await run_turn(NbaState(), [Line(*CALLS["telecom"][1])], llm, settings)  # warm-up: loads the model
    for name, script in CALLS.items():
        state, lines, times = NbaState(), [Line(*row) for row in script], []
        for n, line in enumerate(lines, start=1):
            if line.role != "customer":
                continue
            started = time.perf_counter()
            result = await run_turn(state, lines[:n], llm, settings)
            times.append(time.perf_counter() - started)
            hint = result.hint["text"] if result.hint else "(none)"
            print(f"[{name}] {times[-1]:5.1f}s domain={state.detected} facts={state.facts} hint={hint}")
        print(f"[{name}] average {sum(times) / len(times):.1f}s per turn")


async def main(models: list[str]) -> None:
    for model in models:
        await bench(model)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or ["qwen2.5:7b"]))
