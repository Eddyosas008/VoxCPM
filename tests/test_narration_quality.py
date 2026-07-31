"""Tests for narration.quality — defect detection and targeted re-rolls.

Signals are synthesised rather than generated, so the whole suite runs in
milliseconds with no model: a defect is defined by the shape of the waveform
against its text, and that shape can be built by hand.
"""
from __future__ import annotations

import numpy as np
import pytest

from narration import quality

SR = 24000


def speech(seconds: float, level_db: float = -20.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    """A speech-like signal: a syllable-rate amplitude envelope over noise.

    Not speech, but it shares the two properties every check here reads — a
    steady RMS and an irregular envelope — so it stands in for a clean segment.
    """
    rng = np.random.default_rng(seed)
    samples = int(sr * seconds)
    if samples <= 0:
        return np.zeros(0, dtype=np.float32)
    # Syllable gains drawn at random rather than from a periodic function: a
    # sine envelope would repeat on its own and make every segment look looped.
    syllables = max(2, int(seconds * 6))
    gains = rng.uniform(0.3, 1.0, syllables + 1)
    envelope = np.interp(np.linspace(0.0, syllables, samples), np.arange(syllables + 1), gains)
    signal = rng.normal(0.0, 1.0, samples) * envelope
    signal /= max(float(np.sqrt(np.mean(signal**2))), 1e-12)
    return (signal * 10.0 ** (level_db / 20.0)).astype(np.float32)


def with_edges(body: np.ndarray, lead: float = 0.2, tail: float = 0.35, sr: int = SR) -> np.ndarray:
    """Pad a segment with the silence a well-behaved generation leaves."""
    return np.concatenate(
        [np.zeros(int(sr * lead), dtype=np.float32), body, np.zeros(int(sr * tail), dtype=np.float32)]
    )


def sentence(characters: int) -> str:
    return "a" * characters


# --------------------------------------------------------------------------
# A clean segment
# --------------------------------------------------------------------------


def test_clean_segment_has_no_issues():
    # 140 characters at ~14 char/s is about 10 seconds of speech.
    wav = with_edges(speech(10.0))
    report = quality.inspect_segment(wav, SR, sentence(140))
    assert report.ok, report.describe()
    assert report.severity is None
    assert not report.fatal


def test_clean_segment_reports_its_measurements():
    wav = with_edges(speech(10.0, level_db=-20.0))
    report = quality.inspect_segment(wav, SR, sentence(140))
    assert report.characters == 140
    assert report.duration_sec == pytest.approx(10.55, abs=0.1)
    assert report.chars_per_second == pytest.approx(140 / 10.55, rel=0.05)
    assert report.rms_db == pytest.approx(-20.0, abs=1.5)


def test_slow_and_fast_reading_are_both_accepted():
    # Well below and well above the nominal rate, both legitimate.
    for seconds, characters in ((10.0, 100), (10.0, 300)):
        report = quality.inspect_segment(with_edges(speech(seconds)), SR, sentence(characters))
        assert not report.fatal, f"{characters} chars / {seconds}s: {report.describe()}"


@pytest.mark.parametrize("rate", [15.8, 17.5, 18.8, 19.5, 20.2, 21.1, 23.0, 24.1])
def test_rates_measured_from_the_preset_voices_are_never_flagged(rate):
    """Guards the thresholds against drifting into the engine's real range.

    Every distinct rate the fourteen preset voices produced on the same
    sentence. A change that makes any of them look defective would flag a large
    share of a real book, so it is caught here rather than four hours in.

    The range was unchanged when the set grew from seven voices to fourteen —
    15.8 to 24.1 both times — which is what makes it worth pinning.
    """
    seconds = 8.0
    report = quality.inspect_segment(
        with_edges(speech(seconds)), SR, sentence(int(rate * (seconds + 0.55)))
    )
    assert not report.fatal, f"{rate} char/s flagged: {report.describe()}"


# --------------------------------------------------------------------------
# Fatal defects
# --------------------------------------------------------------------------


def test_empty_audio_is_silent_and_fatal():
    report = quality.inspect_segment(np.zeros(0, dtype=np.float32), SR, sentence(100))
    assert "silent" in report.codes
    assert report.fatal


def test_digital_silence_is_detected():
    report = quality.inspect_segment(np.zeros(SR * 5, dtype=np.float32), SR, sentence(70))
    assert "silent" in report.codes
    assert report.fatal


def test_inaudible_segment_is_detected():
    report = quality.inspect_segment(with_edges(speech(5.0, level_db=-70.0)), SR, sentence(70))
    assert "silent" in report.codes


def test_truncation_is_fatal():
    # A long sentence that came back as one second of audio.
    report = quality.inspect_segment(with_edges(speech(1.0)), SR, sentence(200))
    assert "truncated" in report.codes
    assert report.fatal


def test_runaway_is_fatal():
    # Three characters cannot account for twenty seconds of audio.
    report = quality.inspect_segment(with_edges(speech(20.0)), SR, sentence(3))
    assert "runaway" in report.codes
    assert report.fatal


def test_clipping_is_fatal():
    wav = with_edges(speech(6.0, level_db=-6.0))
    wav[SR : SR + 2000] = 1.0  # a pinned stretch, not a stray sample
    report = quality.inspect_segment(wav, SR, sentence(85))
    assert "clipped" in report.codes
    assert report.fatal


def test_a_single_full_scale_sample_is_not_clipping():
    wav = with_edges(speech(6.0))
    wav[SR] = 1.0
    report = quality.inspect_segment(wav, SR, sentence(85))
    assert "clipped" not in report.codes


def test_empty_text_does_not_trigger_a_rate_defect():
    # Nothing to compare the duration against; dividing by zero characters
    # must not manufacture a truncation.
    report = quality.inspect_segment(with_edges(speech(3.0)), SR, "")
    assert "truncated" not in report.codes
    assert "runaway" not in report.codes


# --------------------------------------------------------------------------
# Suspect defects
# --------------------------------------------------------------------------


def test_internal_gap_is_flagged_as_suspect():
    body = np.concatenate([speech(3.0), np.zeros(int(SR * 2.5), dtype=np.float32), speech(3.0, seed=1)])
    report = quality.inspect_segment(with_edges(body), SR, sentence(120))
    assert "gap" in report.codes
    assert not report.fatal  # worth an ear, not worth an hour of CPU
    assert report.longest_silence_sec == pytest.approx(2.5, abs=0.3)


def test_edge_silence_is_not_an_internal_gap():
    wav = with_edges(speech(8.0), lead=2.0, tail=3.0)
    assert quality.longest_internal_silence_sec(wav, SR) < 0.5
    assert "gap" not in quality.inspect_segment(wav, SR, sentence(110)).codes


def test_normal_sentence_pauses_are_not_gaps():
    body = np.concatenate([speech(4.0), np.zeros(int(SR * 0.5), dtype=np.float32), speech(4.0, seed=2)])
    assert "gap" not in quality.inspect_segment(with_edges(body), SR, sentence(115)).codes


def test_abrupt_end_is_flagged():
    # No trailing silence at all: the waveform stops at speaking level.
    wav = np.concatenate([np.zeros(int(SR * 0.2), dtype=np.float32), speech(8.0)])
    assert quality.ends_abruptly(wav, SR)
    assert "abrupt_end" in quality.inspect_segment(wav, SR, sentence(110)).codes


def test_decayed_end_is_not_abrupt():
    assert not quality.ends_abruptly(with_edges(speech(8.0)), SR)


def test_repeated_phrase_is_flagged_as_looped():
    phrase = speech(2.0, seed=7)
    wav = with_edges(np.concatenate([phrase, phrase, phrase, phrase]))
    assert quality.envelope_repetition(wav, SR) > 0.9
    assert "looped" in quality.inspect_segment(wav, SR, sentence(115)).codes


def test_ordinary_speech_is_not_looped():
    wav = with_edges(speech(10.0, seed=3))
    assert quality.envelope_repetition(wav, SR) < 0.92


def test_short_segment_cannot_be_looped():
    # Too short for a repeat to be measurable — must return 0, not crash.
    assert quality.envelope_repetition(speech(0.4), SR) == 0.0


# --------------------------------------------------------------------------
# Report plumbing
# --------------------------------------------------------------------------


def test_severity_reports_the_worst_issue():
    report = quality.inspect_segment(with_edges(speech(1.0)), SR, sentence(200))
    assert report.severity == quality.FATAL


def test_penalty_prefers_fewer_and_milder_issues():
    clean = quality.inspect_segment(with_edges(speech(10.0)), SR, sentence(140))
    truncated = quality.inspect_segment(with_edges(speech(1.0)), SR, sentence(200))
    assert clean.penalty < truncated.penalty


def test_summarize_counts_by_code():
    good = quality.inspect_segment(with_edges(speech(10.0)), SR, sentence(140))
    bad = quality.inspect_segment(np.zeros(SR, dtype=np.float32), SR, sentence(140))
    summary = quality.summarize([("ch1/seg1", good), ("ch1/seg2", bad)])
    assert summary["segments"] == 2
    assert summary["flagged"] == 1
    assert summary["fatal"] == 1
    assert summary["by_code"]["silent"] == 1
    assert summary["details"][0]["segment"] == "ch1/seg2"


# --------------------------------------------------------------------------
# Retry seeds
# --------------------------------------------------------------------------


def test_retry_seed_is_deterministic():
    assert quality.retry_seed(1234, 1, "bonjour") == quality.retry_seed(1234, 1, "bonjour")


def test_retry_seed_differs_per_attempt_and_text():
    assert quality.retry_seed(1234, 1, "bonjour") != quality.retry_seed(1234, 2, "bonjour")
    assert quality.retry_seed(1234, 1, "bonjour") != quality.retry_seed(1234, 1, "bonsoir")
    assert quality.retry_seed(1234, 1, "bonjour") != 1234


def test_retry_seed_stays_in_engine_range():
    for attempt in range(1, 20):
        value = quality.retry_seed(2**31, attempt, "x")
        assert 0 <= value < 2**32


def test_retry_seed_of_none_is_none():
    # The engine already randomises; asking again is enough.
    assert quality.retry_seed(None, 1, "bonjour") is None


# --------------------------------------------------------------------------
# render_checked
# --------------------------------------------------------------------------


def _good(_seed=None):
    return SR, with_edges(speech(10.0))


def _truncated(_seed=None):
    return SR, with_edges(speech(0.5))


def test_render_checked_keeps_a_good_first_take():
    calls = []

    def render(seed):
        calls.append(seed)
        return _good(seed)

    result = quality.render_checked(sentence(140), render, base_seed=99, max_attempts=3)
    assert result.report.ok
    assert result.attempts == 1
    assert calls == [99]  # no re-roll when nothing is wrong
    assert not result.repaired


def test_render_checked_rerolls_a_fatal_take():
    def render(seed):
        return _truncated(seed) if seed == 99 else _good(seed)

    result = quality.render_checked(sentence(140), render, base_seed=99, max_attempts=3)
    assert result.attempts == 2
    assert not result.report.fatal
    assert result.repaired
    assert result.seed != 99
    assert [r.codes for r in result.rejected] == [("truncated",)]


def test_render_checked_stops_at_max_attempts():
    calls = []

    def render(seed):
        calls.append(seed)
        return _truncated(seed)

    result = quality.render_checked(sentence(140), render, base_seed=1, max_attempts=3)
    assert len(calls) == 3
    assert result.attempts == 3
    assert result.unrepairable


def test_render_checked_returns_the_best_attempt_not_the_last():
    # First take is merely truncated, second is silent — the first must win.
    takes = [(SR, with_edges(speech(3.0))), (SR, np.zeros(SR * 3, dtype=np.float32))]

    def render(_seed):
        return takes.pop(0)

    result = quality.render_checked(sentence(200), render, base_seed=5, max_attempts=2)
    assert "silent" not in result.report.codes
    assert result.seed == 5
    assert quality.audio_tools.peak_db(result.wav) > -50.0


def test_render_checked_survives_an_engine_crash_on_one_seed():
    def render(seed):
        if seed == 42:
            raise RuntimeError("engine died on this seed")
        return _good(seed)

    result = quality.render_checked(sentence(140), render, base_seed=42, max_attempts=3)
    assert result.report.ok
    assert result.attempts == 2  # the crashed call counts as an attempt


def test_render_checked_reraises_when_every_attempt_crashes():
    def render(_seed):
        raise RuntimeError("engine is down")

    with pytest.raises(RuntimeError, match="engine is down"):
        quality.render_checked(sentence(140), render, base_seed=1, max_attempts=2)


def test_render_checked_reports_each_attempt():
    seen = []

    def render(seed):
        return _truncated(seed) if seed == 7 else _good(seed)

    quality.render_checked(
        sentence(140),
        render,
        base_seed=7,
        max_attempts=2,
        on_attempt=lambda index, report: seen.append((index, report.codes)),
    )
    assert seen[0] == (0, ("truncated",))
    assert seen[1][0] == 1


def test_render_checked_never_calls_render_more_than_once_when_ok():
    calls = []
    quality.render_checked(
        sentence(140), lambda seed: (calls.append(seed), _good(seed))[1], base_seed=None, max_attempts=5
    )
    assert len(calls) == 1
