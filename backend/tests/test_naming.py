"""Naming the voices in the call audio from what people say."""
from app.naming import addressed, answered, decide, introduced, name_votes
from app.sessions import Utterance, is_call_speaker


def line(speaker: str, text: str, start: float, end: float | None = None) -> Utterance:
    name = {"u1": "Aklavya"}.get(speaker, "Computer")
    return Utterance(f"{speaker}{start}", speaker, name, text, start, end if end is not None else start + 3)


def test_clues_in_a_line():
    assert addressed("Priya, can you retest it?") == "Priya"
    assert addressed("Okay Balram, you own the coupon fix.") == "Balram"
    assert addressed("Okay, please create a Jira ticket") is None
    assert addressed("Thursday, we release.") is None
    assert answered("Will do, Vikram. I'll send the update.") == "Vikram"
    assert answered("Thanks Sarah.") == "Sarah"
    assert answered("Thanks everyone.") is None
    assert introduced("Hi, Priya here.") == "Priya"
    assert introduced("This is Balram from QA") == "Balram"


def test_the_sprint_call_names_every_voice():
    lines = [
        line("u1", "Hi everyone. Balram, Priya, Sara and Vikram are here, so let's start.", 0),
        line("u1-call", "Thanks Aklavya. The release is on Friday.", 4),
        line("u1-call2", "The first bug: the coupon code is applied twice.", 8),
        line("u1", "Balram, you own the coupon fix. Can you have it done by Wednesday?", 12),
        line("u1-call3", "Yes Aklavya, I'll fix the coupon bug by Wednesday evening.", 16),
        line("u1", "Sara, please write the pagination design by Tuesday.", 20),
        line("u1-call4", "Sure, I'll share the pagination design on Tuesday.", 24),
        line("u1", "Priya, you retest both bugs on Thursday evening.", 28),
        line("u1-call2", "Got it. I'll retest Thursday evening.", 32),
        line("u1-call", "Good. Aklavya, please send me a status update on Thursday.", 36),
        line("u1", "Will do, Vikram. I'll send the update on Thursday.", 40),
    ]
    votes = name_votes(lines, is_call_speaker)
    names = decide(votes, {"u1-call", "u1-call2", "u1-call3", "u1-call4"}, taken={"Aklavya"})
    assert names == {"u1-call": "Vikram", "u1-call2": "Priya", "u1-call3": "Balram", "u1-call4": "Sara"}


def test_my_own_name_and_names_taken_are_never_given():
    lines = [
        line("u1-call", "Aklavya, please send me the update.", 0),
        line("u1-call2", "Sure.", 4),
        line("u1", "Priya, can you check?", 8),
        line("u1-call2", "Yes.", 12),
    ]
    names = decide(name_votes(lines, is_call_speaker), {"u1-call2"}, taken={"Aklavya", "Priya"})
    assert names == {}


def test_a_voice_with_competing_clues_waits_and_a_name_goes_to_one_voice_only():
    lines = [
        line("u1", "Priya, can you check?", 0),
        line("u1-call", "Yes.", 4),
        line("u1", "Sara, and you?", 8),
        line("u1-call", "Also yes.", 12),
        line("u1", "Priya, one more thing?", 16),
        line("u1-call2", "Sure.", 20),
    ]
    votes = name_votes(lines, is_call_speaker)
    # u1-call has one vote each for Priya and Sara: not sure. u1-call2 has one clear vote for Priya.
    assert decide(votes, {"u1-call", "u1-call2"}, taken=set()) == {"u1-call2": "Priya"}


def test_two_votes_win_over_one():
    lines = [
        line("u1", "Priya, can you check?", 0),
        line("u1-call", "Yes.", 4),
        line("u1", "Priya, and the login page?", 8),
        line("u1-call", "That too.", 12),
        line("u1", "Sara, what do you think?", 16),
        line("u1-call", "Fine by me.", 20),
    ]
    assert decide(name_votes(lines, is_call_speaker), {"u1-call"}, taken=set()) == {"u1-call": "Priya"}


def test_late_or_unrelated_lines_give_no_vote():
    lines = [
        line("u1", "Priya, can you check?", 0, 2),
        line("u1-call", "Hello?", 30),  # 28 s later: not an answer
        line("u1-call2", "Hi, Balram here.", 40),
    ]
    votes = name_votes(lines, is_call_speaker)
    assert "u1-call" not in votes
    assert decide(votes, {"u1-call", "u1-call2"}, taken=set()) == {"u1-call2": "Balram"}


def test_a_misheard_spelling_of_a_taken_name_is_not_given():
    from app.naming import clashes

    assert clashes("Aquavia", {"Aklavya"}) and clashes("Aklavaya", {"Aklavya"}) and clashes("Sara", {"Sarah"})
    assert not clashes("Vikram", {"Aklavya", "Balram"}) and not clashes("Priya", {"Aklavya"})
    lines = [line("u1-call", "Hi.", 0), line("u1-call2", "Good. Aquavia, please send me the update.", 4)]
    # "Good. Aquavia" answers u1-call by its pattern, but Aquavia is how Aklavya was heard.
    assert decide(name_votes(lines, is_call_speaker), {"u1-call"}, taken={"Aklavya"}) == {}
