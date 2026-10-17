"""Naming the voices in the call audio: the model lists the people and picks who each voice is; the code only checks
the answers against the call."""
from app.naming import (NAMES_PROMPT, UNKNOWN, VOICE_PROMPT, candidates, name_voices, names_input, neutral_labels,
                        settle, voice_input)
from app.sessions import Utterance, is_call_speaker


def line(speaker: str, name: str, text: str, start: float) -> Utterance:
    return Utterance(f"{speaker}{start}", speaker, name, text, start, start + 3)


ESCALATION = [
    line("u1", "Aklavya", "Hi Neha, Rohan and Karan, thanks for joining.", 0),
    line("u1-call", "Computer", "Thanks Aklavya. Apex can't export the monthly report.", 4),
    line("u1-call2", "Computer 2", "The export times out, and it fails when the file is ready.", 8),
    line("u1", "Aklavya", "Rohan, can you move it to a background job by Thursday?", 12),
    line("u1-call2", "Computer 2", "Yes, I'll move the export to a background job.", 16),
    line("u1", "Aklavya", "Karan, please send Meera a status update by tomorrow.", 20),
    line("u1-call", "Computer", "Will do, Aklavya. I'll send it before noon.", 24),
]
UNNAMED = {"u1-call": "Computer", "u1-call2": "Computer 2"}


class FakeLLM:
    """Answers the name list with `names` and each voice with `picks[label]`."""

    def __init__(self, names=None, picks=None, fail=False):
        self.names, self.picks, self.fail, self.calls = names or [], picks or {}, fail, []

    async def json(self, system, user, schema):
        self.calls.append((system, user, schema))
        if self.fail:
            raise RuntimeError("model down")
        if system == NAMES_PROMPT:
            return {"names": self.names}, "fake"
        label = user.split("\n", 1)[0].removeprefix("Unknown voice: ")
        return {"why": "", "name": self.picks.get(label, UNKNOWN)}, "fake"


def test_listed_names_are_kept_only_when_written_as_names_in_the_call_and_free():
    """Seen live: "...when it's ready" became a voice named "Ready" under phrase rules."""
    answer = {"names": ["Aklavya", "Neha", "Rohan", "ready", "Ready", "Vikram", "Karan", "Karan", "Aquavia"]}
    assert candidates(answer, ESCALATION, {"Aklavya"}) == ["Neha", "Rohan", "Karan"]


def test_a_name_two_voices_pick_waits_and_unknown_names_nobody():
    picks = {"a": "Neha", "b": "Neha", "c": "Rohan", "d": UNKNOWN}
    assert settle(picks) == {"c": "Rohan"}


def test_the_model_reads_unknown_voices_as_neutral_speakers():
    labels = neutral_labels(ESCALATION, is_call_speaker)
    assert labels == {"u1-call": "Speaker A", "u1-call2": "Speaker B"}
    text = names_input(ESCALATION, labels)
    assert "Speaker A: Thanks Aklavya.\nSpeaker A: Apex can't export the monthly report." in text
    assert "Computer" not in text
    asked = voice_input(ESCALATION, "u1-call2", ["Neha", "Rohan", "Karan"], labels)
    assert asked.startswith("Unknown voice: Speaker B\nPeople: Neha, Rohan, Karan\n\n")
    assert "Aklavya: Rohan, can you move it to a background job by Thursday?\nSpeaker B: Yes, I'll move" in asked
    assert asked.endswith("Which of the people is Speaker B?")


async def test_voices_get_the_names_the_model_picks_from_the_list():
    llm = FakeLLM(names=["Aklavya", "Neha", "Rohan", "Karan"], picks={"Speaker A": "Karan", "Speaker B": "Rohan"})
    assert await name_voices(ESCALATION, UNNAMED, {"Aklavya"}, llm) == {"u1-call": "Karan", "u1-call2": "Rohan"}
    voice_calls = [c for c in llm.calls if c[0] == VOICE_PROMPT]
    assert len(voice_calls) == 2
    assert voice_calls[0][2]["properties"]["name"]["enum"] == ["Neha", "Rohan", "Karan", UNKNOWN]


async def test_no_names_in_the_call_or_a_failing_model_names_nobody():
    llm = FakeLLM(names=[])
    assert await name_voices(ESCALATION, UNNAMED, set(), llm) == {} and len(llm.calls) == 1
    assert await name_voices(ESCALATION, UNNAMED, set(), FakeLLM(fail=True)) == {}
    idle = FakeLLM()
    assert await name_voices(ESCALATION, {}, set(), idle) == {} and idle.calls == []


def test_the_input_keeps_the_latest_lines_of_a_long_call():
    lines = [line("u1", "Aklavya", f"Line {i}.", i) for i in range(300)]
    text = names_input(lines)
    assert "Aklavya: Line 299." in text and "Aklavya: Line 99." not in text
