from app.segmenter import CANCEL, FINAL, PARTIAL, Segmenter
from app.stt.base import clean_transcript
from app.vad import EnergyVAD
from tests.conftest import SR, chunks, silence, tone


def make(**kwargs) -> Segmenter:
    kwargs.setdefault("partials_enabled", False)
    return Segmenter(EnergyVAD(-45.0), sample_rate=SR, **kwargs)


def feed_all(segmenter: Segmenter, pcm: bytes, start: float = 100.0):
    out, ts = [], start
    for chunk in chunks(pcm):
        out.extend(segmenter.feed(chunk, ts))
        ts += len(chunk) / 2 / SR
    return out


def test_silence_only_produces_nothing():
    seg = make()
    assert feed_all(seg, silence(3)) == []
    assert seg.flush() == []


def test_two_sentences_split_by_silence():
    seg = make()
    pcm = silence(0.5) + tone(1.0) + silence(1.0) + tone(1.5) + silence(1.0)
    finals = [s for s in feed_all(seg, pcm) if s.kind == FINAL]
    assert len(finals) == 2
    first, second = finals
    # first utterance starts about 0.5s in (minus ~200ms pre-roll), relative to the stream start of 100.0
    assert 100.2 <= first.start_ts <= 100.5
    assert 101.5 <= first.end_ts <= 102.3
    assert second.start_ts > first.end_ts
    assert first.utterance_id != second.utterance_id
    # audio covers the speech plus pre-roll and trailing silence
    assert len(first.pcm) / 2 / SR >= 1.0


def test_short_click_is_ignored():
    seg = make(min_speech_ms=250)
    assert [s for s in feed_all(seg, silence(0.5) + tone(0.1) + silence(1.0)) if s.kind == FINAL] == []


def test_flush_closes_open_utterance():
    seg = make()
    assert feed_all(seg, tone(1.0)) == []
    finals = seg.flush()
    assert [s.kind for s in finals] == [FINAL]


def test_long_speech_is_cut_at_max_length():
    seg = make(max_utterance_ms=2000)
    finals = [s for s in feed_all(seg, tone(5.0) + silence(1.0)) if s.kind == FINAL]
    assert len(finals) == 3
    assert all(len(s.pcm) / 2 / SR <= 2.05 for s in finals)


def test_partials_emitted_while_speaking_then_final():
    seg = make(partials_enabled=True, partial_interval_ms=500)
    out = feed_all(seg, tone(2.0) + silence(1.0))
    kinds = [s.kind for s in out]
    assert kinds.count(PARTIAL) >= 3 and kinds[-1] == FINAL
    assert len({s.utterance_id for s in out}) == 1


def test_noise_after_partial_is_cancelled():
    # quiet tone barely over threshold for a moment -> partial, but not enough speech for a final
    seg = make(partials_enabled=True, partial_interval_ms=100, min_speech_ms=500)
    out = feed_all(seg, tone(0.2) + silence(1.0))
    assert [s.kind for s in out][-1] == CANCEL


def test_clean_transcript():
    assert clean_transcript("[NOISE]") == ""
    assert clean_transcript("  hello   [NOISE] world ") == "hello world"
    assert clean_transcript("no no no no no no") == "no no no"
    assert clean_transcript("sooooo 1111111") == "sooo 1111111"
