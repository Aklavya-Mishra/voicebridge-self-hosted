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


def test_call_voices_are_computer_then_computer_n_and_can_be_renamed_before_a_saved_line():
    from app.sessions import call_speaker_id, call_speaker_name, is_call_speaker

    assert call_speaker_id("u1") == "u1-call" and call_speaker_id("u1", 3) == "u1-call3"
    assert call_speaker_name("u1-call") == "Computer" and call_speaker_name("u1-call2") == "Computer 2"
    assert is_call_speaker("u1-call") and is_call_speaker("u1-call4") and not is_call_speaker("u1-room2")
    store = SessionStore()
    code = store.create({"id": "u1", "name": "Aklavya"}).code
    assert store.rename_speaker(code, "u1-call2", "Balram") == "Computer 2"
    assert store.speaker_name(code, "u1-call2", "?") == "Balram"


def _handlers(settings=None):
    from app.config import Settings
    from app.socket_events import SocketHandlers

    class Sio:
        def __init__(self):
            self.emitted = []

        def on(self, name, handler):
            pass

        async def emit(self, event, payload, room=None, to=None):
            self.emitted.append((event, payload))

    store = SessionStore()
    code = store.create({"id": "u1", "name": "Aklavya"}).code
    sio = Sio()
    return SocketHandlers(sio, store, settings or Settings(jwt_secret="t"), stt=None, llm=None), store, code, sio


def test_call_audio_tells_voices_apart_unless_turned_off():
    from app.config import Settings

    handlers, _, code, _ = _handlers()
    made = []
    handlers._new_stream = lambda c, speaker, speaker_for=None: made.append((speaker["id"], speaker_for)) or "s"
    handlers._stream_for(code, {"id": "u1-call", "name": "Computer"})
    handlers._stream_for(code, {"id": "u1", "name": "Aklavya"})
    (call_id, voices), (mic_id, none) = made
    assert call_id == "u1-call" and mic_id == "u1" and none is None
    assert voices(0) == {"id": "u1-call", "name": "Computer"}
    assert voices(1) == {"id": "u1-call2", "name": "Computer 2"}
    off, _, code, _ = _handlers(Settings(jwt_secret="t", call_diarize=False))
    made.clear()
    off._new_stream = lambda c, speaker, speaker_for=None: made.append(speaker_for) or "s"
    off._stream_for(code, {"id": "u1-call", "name": "Computer"})
    assert made == [None]


def say(store, code, speaker, name, text, start):
    store.add_utterance(code, Utterance(f"{speaker}{start}", speaker, name, text, start, start + 3))


async def test_voices_are_named_live_and_a_name_typed_by_hand_is_kept():
    handlers, store, code, sio = _handlers()
    say(store, code, "u1", "Aklavya", "Priya, can you retest the login page?", 0)
    say(store, code, "u1-call2", "Computer 2", "Sure, I'll retest it today.", 4)
    await handlers._auto_name(code)
    assert store.speaker_name(code, "u1-call2", "?") == "Priya"
    assert [u.speaker_name for u in store.get(code).transcript] == ["Aklavya", "Priya"]
    assert ("speaker_renamed", {"code": code, "speaker_id": "u1-call2", "name": "Priya"}) in sio.emitted
    # A voice renamed by hand is never named again by the clues.
    say(store, code, "u1-call", "Computer", "And the payments bug?", 8)
    store.rename_speaker(code, "u1-call", "Rohan")
    handlers._named_by_hand.setdefault(code, set()).add("u1-call")
    say(store, code, "u1", "Aklavya", "Thanks Vikram.", 12)
    await handlers._auto_name(code)
    assert store.speaker_name(code, "u1-call", "?") == "Rohan"


def test_only_the_call_audio_lines_are_swapped():
    store = SessionStore()
    code = store.create({"id": "u1", "name": "Aklavya"}).code
    say(store, code, "u1", "Aklavya", "mine", 1)
    say(store, code, "u1-call", "Computer", "old", 2)
    new = Utterance("n1", "u1-call2", "Computer 2", "new", 2.5, 3)
    store.replace_utterances(code, 0, 10, [new], only=lambda u: u.speaker_id.startswith("u1-call"))
    assert [u.text for u in store.get(code).transcript] == ["mine", "new"]


async def test_call_recheck_rebuilds_only_the_call_voices_and_keeps_names():
    from app.room import Piece, recheck
    from app.sessions import call_speaker_id, call_speaker_name

    store = SessionStore()
    code = store.create({"id": "u1", "name": "Aklavya"}).code
    say(store, code, "u1", "Aklavya", "Priya, status?", 100.0)
    say(store, code, "u1-call", "Computer", "done and vikram here", 104.0)
    store.rename_speaker(code, "u1-call", "Priya")

    class Stt:
        async def diarize(self, pcm, rate):
            return [
                {"start": 4.0, "end": 4.4, "word": "done", "speaker": 0},
                {"start": 4.5, "end": 4.9, "word": "and", "speaker": 0},
                {"start": 6.0, "end": 6.4, "word": "vikram", "speaker": 1},
                {"start": 6.5, "end": 6.9, "word": "here", "speaker": 1},
            ]

    changed = await recheck(
        store, code, "u1", [Piece(100.0, bytearray(16000 * 2 * 10))], Stt(), 16000, 1.0, False,
        speaker_id=call_speaker_id, default_name=call_speaker_name,
        only=lambda u: u.speaker_id.startswith("u1-call"),
    )
    assert changed
    lines = [(u.speaker_id, u.speaker_name, u.text) for u in store.get(code).transcript]
    assert lines == [("u1", "Aklavya", "Priya, status?"), ("u1-call", "Priya", "done and"),
                     ("u1-call2", "Computer 2", "vikram here")]


def test_long_silences_are_cut_before_the_recheck_and_times_stay_right():
    import numpy as np

    from app.room import Piece, _at, drop_silence

    rate = 16000
    tone = (np.ones(rate // 2) * 3000).astype(np.int16)  # 0.5 s heard
    pcm = np.concatenate([np.zeros(rate * 2, np.int16), tone, np.zeros(rate * 3, np.int16), tone]).tobytes()
    parts = drop_silence([Piece(100.0, bytearray(pcm))], rate)
    assert len(parts) == 2
    assert abs(parts[0].start_ts - (102.0 - 0.15)) < 1e-3 and abs(parts[1].start_ts - (105.5 - 0.15)) < 1e-3
    # 0.8 s into the cut audio is 0.15 s into the second heard part
    assert abs(_at(parts, rate, 0.8 + 0.15) - (105.5)) < 0.01
    assert drop_silence([Piece(0.0, bytearray(np.zeros(rate, np.int16).tobytes()))], rate) == []


def test_a_word_starting_just_before_its_piece_stays_in_it():
    from app.room import Piece, _word_times

    rate = 16000
    pieces = [Piece(100.0, bytearray(rate * 2 * 2)), Piece(150.0, bytearray(rate * 2 * 2))]  # 2 s each
    # "Yes" starts 0.05 s before the second piece (in the cut audio) but is mostly in it
    times = _word_times(pieces, rate, 1.95, 2.3)
    assert abs(times["ts"] - 150.0) < 1e-6 and abs(times["end_ts"] - 150.3) < 1e-6
    assert abs(_word_times(pieces, rate, 0.5, 0.8)["ts"] - 100.5) < 1e-6
