"""In-room mode: the meeting audio kept for the re-check, matching voices to speakers, rebuilding the transcript."""
import pytest

from app.analysis import analyse_session
from app.llm import LlmChain
from app.room import Piece, RoomRecorder, build_lines, recheck, speakers_by_label
from app.sessions import SessionStore, Utterance, room_speaker_name

SR = 16000
ME = {"id": "u", "name": "Aklavya"}
T0 = 1000.0


def second(n: float = 1.0) -> bytes:
    return b"\x00\x00" * int(SR * n)


def test_recorder_keeps_continuous_audio_as_one_piece_and_a_gap_starts_another():
    rec = RoomRecorder(SR)
    rec.add("C", second(), T0)
    rec.add("C", second(), T0 + 1.0)  # continues
    rec.add("C", second(), T0 + 10.0)  # muted for a while
    pieces = rec.take("C")
    assert [(p.start_ts, len(p.pcm)) for p in pieces] == [(T0, 2 * SR * 2), (T0 + 10.0, SR * 2)]
    assert rec.take("C") == []  # taken: forgotten


def w(ts: float, word: str, speaker: int, end: float | None = None) -> dict:
    return {"ts": ts, "end_ts": end if end is not None else ts + 0.3, "word": word, "speaker": speaker}


def line(speaker_id: str, start: float, end: float, text: str = "x") -> Utterance:
    return Utterance(f"{speaker_id}-{start}", speaker_id, speaker_id, text, start, end)


def test_voices_go_to_the_live_speaker_sharing_most_of_their_words():
    # Live put the second person's first line under speaker 1 (the cold start); most of their words are speaker 2.
    live = [line("u-room1", 0, 5), line("u-room1", 6, 8), line("u-room2", 10, 14), line("u-room1", 15, 18)]
    words = [w(1, "a", 0), w(2, "b", 0), w(6.5, "c", 1), w(11, "d", 1), w(12, "e", 1), w(16, "f", 0)]
    assert speakers_by_label(words, live) == {0: "u-room1", 1: "u-room2"}


def test_lines_are_turns_of_one_voice_and_a_long_silence_splits_them():
    speakers = {0: {"id": "s1", "name": "Vikram"}, 1: {"id": "s2", "name": "Rohan"}}
    words = [w(0, "Six", 0), w(0.4, "percent.", 0), w(1.2, "Okay.", 1), w(5.0, "Sure,", 1), w(5.4, "fine.", 1)]
    lines = build_lines(words, speakers, gap_s=3.0, masking=True)
    assert [(u.speaker_name, u.text) for u in lines] == [
        ("Vikram", "Six percent."), ("Rohan", "Okay."), ("Rohan", "Sure, fine."),
    ]
    assert lines[0].start_ts == 0 and lines[0].end_ts == pytest.approx(0.7)


def test_rebuilt_lines_are_masked():
    lines = build_lines([w(0, "call", 0), w(0.5, "9876543210", 0)], {0: {"id": "s", "name": "S"}}, 3.0, True)
    assert lines[0].text == "call xxxxx"


class FakeDiarizer:
    def __init__(self, words=None, fail=False):
        self.words, self.fail, self.calls = words or [], fail, []

    async def diarize(self, pcm: bytes, sample_rate: int) -> list[dict]:
        self.calls.append(len(pcm))
        if self.fail:
            raise RuntimeError("deepgram down")
        return self.words


def meeting(store: SessionStore) -> str:
    code = store.create(ME, mode="room").code
    # Live: the cold start put Rohan's reply under Speaker 1.
    for u in (
        Utterance("l1", "u-room1", "Speaker 1", "Your increment is six percent. Six percent?", T0 + 0.1, T0 + 3.0),
        Utterance("l2", "u-room2", "Speaker 2", "I was expecting more.", T0 + 4.0, T0 + 5.5),
    ):
        store.add_utterance(code, u)
    store.rename_speaker(code, "u-room1", "Vikram")
    return code


async def test_recheck_rebuilds_the_span_with_the_better_labels_and_keeps_names(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "db.sqlite"))
    code = meeting(store)
    words = [
        {"start": 0.1, "end": 0.5, "word": "Your", "speaker": 0},
        {"start": 0.6, "end": 1.9, "word": "increment is six percent.", "speaker": 0},
        {"start": 2.2, "end": 3.0, "word": "Six percent?", "speaker": 1},
        {"start": 4.0, "end": 5.5, "word": "I was expecting more.", "speaker": 1},
    ]
    stt = FakeDiarizer(words)
    pieces = [Piece(T0, bytearray(second(6)))]

    assert await recheck(store, code, "u", pieces, stt, SR, 3.0, True)

    lines = [(u.speaker_name, u.text) for u in store.get(code).transcript]
    assert lines == [("Vikram", "Your increment is six percent."), ("Speaker 2", "Six percent? I was expecting more.")]
    assert store.get(code).transcript[1].start_ts == pytest.approx(T0 + 2.2)
    # Saved as well: the old lines are gone after a restart.
    reloaded = SessionStore(db_path=str(tmp_path / "db.sqlite")).get(code)
    assert [u.text for u in reloaded.transcript] == ["Your increment is six percent.", "Six percent? I was expecting more."]


async def test_recheck_times_follow_each_piece_of_audio():
    store = SessionStore()
    code = store.create(ME, mode="room").code
    store.add_utterance(code, Utterance("l1", "u-room1", "Speaker 1", "before", T0, T0 + 0.5))
    store.add_utterance(code, Utterance("l2", "u-room1", "Speaker 1", "after", T0 + 30.2, T0 + 30.8))
    # 1 s recorded at T0, then (after a mute) 1 s at T0 + 30: Deepgram hears 2 s in a row.
    words = [{"start": 0.1, "end": 0.5, "word": "before", "speaker": 0}, {"start": 1.2, "end": 1.8, "word": "after", "speaker": 0}]
    pieces = [Piece(T0, bytearray(second())), Piece(T0 + 30.0, bytearray(second()))]
    assert await recheck(store, code, "u", pieces, FakeDiarizer(words), SR, 3.0, True)
    assert [round(u.start_ts - T0, 1) for u in store.get(code).transcript] == [0.1, 30.2]


@pytest.mark.parametrize("stt", [FakeDiarizer(fail=True), FakeDiarizer(words=[]), object()])
async def test_recheck_keeps_the_live_transcript_when_it_cannot_do_better(stt):
    store = SessionStore()
    code = meeting(store)
    before = [u.text for u in store.get(code).transcript]
    assert not await recheck(store, code, "u", [Piece(T0, bytearray(second(6)))], stt, SR, 3.0, True)
    assert [u.text for u in store.get(code).transcript] == before


async def test_a_voice_live_never_heard_becomes_the_next_speaker():
    store = SessionStore()
    code = store.create(ME, mode="room").code
    store.add_utterance(code, Utterance("l1", "u-room1", "Speaker 1", "hello there", T0, T0 + 1))
    words = [{"start": 0.0, "end": 1.0, "word": "hello there", "speaker": 0}, {"start": 1.5, "end": 2.0, "word": "hi", "speaker": 1}]
    assert await recheck(store, code, "u", [Piece(T0, bytearray(second(3)))], FakeDiarizer(words), SR, 3.0, True)
    assert [(u.speaker_id, u.speaker_name) for u in store.get(code).transcript] == [
        ("u-room1", "Speaker 1"), ("u-room2", "Speaker 2"),
    ]


def test_room_speakers_are_named_by_number_and_can_be_renamed_before_they_have_a_saved_line():
    store = SessionStore()
    code = store.create(ME, mode="room").code
    assert room_speaker_name("u-room2") == "Speaker 2"
    assert store.rename_speaker(code, "u-room2", "Rohan") == "Speaker 2"
    assert store.speaker_name(code, "u-room2", "") == "Rohan"


def test_mode_is_saved_and_defaults_to_call(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "db.sqlite"))
    room, call = store.create(ME, mode="room").code, store.create(ME).code
    reloaded = SessionStore(db_path=str(tmp_path / "db.sqlite"))
    assert reloaded.get(room).mode == "room" and reloaded.get(call).mode == "call"
    assert reloaded.get(room).to_dict()["mode"] == "room"


class MinutesLLM:
    name = "fake"

    def __init__(self, result: dict):
        self.result, self.prompts = result, []

    async def json(self, system, user, schema):
        self.prompts.append((system, user, schema))
        return self.result


async def test_a_meeting_gets_minutes_with_attendees_and_no_sentiment():
    store = SessionStore()
    code = store.create(ME, mode="room").code
    store.add_utterance(code, Utterance("1", "u-room1", "Vikram", "Your increment is six percent.", T0, T0 + 2))
    store.add_utterance(code, Utterance("2", "u-room2", "Rohan", "Okay.", T0 + 3, T0 + 4))
    store.set_tickets(code, [{"id": "t", "status": "created", "key": "SCRUM-9", "title": "Analytics handover", "ts": 1}])
    llm = MinutesLLM(
        {
            "summary": "Vikram gave Rohan a six percent increment.",
            "discussion_points": ["Rohan listed his achievements."],
            "decisions": ["Increment stays at six percent."],
            "action_items": ["Vikram will send revised goals by Friday", "Rohan will create a Jira ticket", "none"],
            "open_issues": ["Rohan feels his work deserved more."],
            "pii": [],
        }
    )
    await analyse_session(store, code, LlmChain([llm]))

    summary = store.get(code).summary
    assert len(llm.prompts) == 1 and "minutes" in llm.prompts[0][0]  # one request; no sentiment request
    assert summary["kind"] == "minutes" and summary["status"] == "ready"
    assert summary["attendees"] == ["Vikram", "Rohan"]
    assert summary["action_items"] == ["Vikram will send revised goals by Friday"]  # the ticket is done, not a step
    assert summary["done"] == ["Jira ticket SCRUM-9 created: Analytics handover"]
    assert summary["open_issues"] == ["Rohan feels his work deserved more."]
    assert "sentiment" not in summary
