import numpy as np
import pytest

from radiosonify._events import (
    _add_event_layer,
    _detect_events,
    _render_water_drop,
)
from radiosonify._perceptual_config import PERCEPTUAL_CHOICES


@pytest.mark.parametrize("event_voice", PERCEPTUAL_CHOICES["event_voice"])
def test_each_configured_event_voice_has_a_working_renderer(event_voice):
    audio = np.zeros(16)
    rendered = _add_event_layer(
        audio,
        event_voice=event_voice,
        salience=np.zeros((2, 1)),
        frequencies=np.array([300.0]),
        compensation_gains=np.ones(1),
        sr=8_000,
        duration=0.1,
        threshold=0.5,
        max_events_per_second=1,
        decay_ms=70,
        level_db=-20,
        harmonic_limit_hz=2_000,
    )

    np.testing.assert_array_equal(rendered, audio)


def test_event_detector_keeps_one_strongest_local_peak_per_rate_bucket():
    salience = np.zeros((10, 5))
    salience[1, 1] = 0.7
    salience[3, 3] = 0.9
    salience[6, 2] = 0.8
    salience[8, 4] = 0.6

    events = _detect_events(
        salience,
        duration=1.0,
        threshold=0.5,
        max_events_per_second=2.0,
    )

    assert [(event.frame, event.band, event.strength) for event in events] == [
        (3, 3, 0.9),
        (6, 2, 0.8),
    ]


def test_event_detector_collapses_adjacent_peaks_and_obeys_zero_budget():
    salience = np.zeros((8, 4))
    salience[3, 1] = 1.0
    salience[3, 2] = 0.8

    events = _detect_events(
        salience,
        duration=1.0,
        threshold=0.5,
        max_events_per_second=8.0,
    )

    assert [(event.frame, event.band) for event in events] == [(3, 1)]
    assert (
        _detect_events(
            salience,
            duration=0.1,
            threshold=0.5,
            max_events_per_second=5.0,
        )
        == ()
    )


def test_event_detector_selects_one_representative_for_a_flat_peak():
    salience = np.zeros((12, 6))
    salience[3:6, 2:5] = 0.8

    events = _detect_events(
        salience,
        duration=2.0,
        threshold=0.5,
        max_events_per_second=6.0,
    )

    assert [(event.frame, event.band) for event in events] == [(3, 2)]


def test_water_drop_is_finite_bounded_and_decays():
    drop = _render_water_drop(
        sr=16_000,
        frequency=900.0,
        decay_ms=160.0,
        harmonic_limit_hz=3_500.0,
    )

    assert drop.shape == (2_560,)
    assert np.all(np.isfinite(drop))
    assert np.max(np.abs(drop)) <= 1.2
    assert np.sqrt(np.mean(drop[-256:] ** 2)) < np.sqrt(np.mean(drop[:256] ** 2)) * 0.03


def test_water_drop_is_deterministic_with_coordinate_driven_variation():
    settings = {
        "sr": 48_000,
        "frequency": 900.0,
        "decay_ms": 160.0,
        "harmonic_limit_hz": 3_500.0,
    }

    first = _render_water_drop(**settings, variation_key=42)
    repeated = _render_water_drop(**settings, variation_key=42)
    neighboring = _render_water_drop(**settings, variation_key=43)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, neighboring)


def test_water_drop_has_a_quiet_impact_before_the_dominant_bubble_packet():
    sr = 48_000
    frequency = 900.0
    drop = _render_water_drop(
        sr=sr,
        frequency=frequency,
        decay_ms=160.0,
        harmonic_limit_hz=3_500.0,
        variation_key=0,
    )

    impact = drop[: int(0.003 * sr)]
    bubble = drop[int(0.005 * sr) : int(0.060 * sr)]
    assert np.max(np.abs(impact)) > 1e-4
    assert np.max(np.abs(impact)) < np.max(np.abs(bubble)) * 0.2

    windowed = bubble * np.hanning(len(bubble))
    spectrum = np.abs(np.fft.rfft(windowed))
    bins = np.fft.rfftfreq(len(windowed), 1.0 / sr)
    dominant_frequency = float(bins[int(np.argmax(spectrum))])
    assert dominant_frequency == pytest.approx(frequency, abs=30.0)


def test_none_event_layer_is_identity_and_water_drop_changes_owned_audio():
    audio = np.zeros(8_000)
    salience = np.zeros((20, 3))
    salience[5, 1] = 1.0
    common = {
        "salience": salience,
        "frequencies": np.array([300.0, 600.0, 1_200.0]),
        "compensation_gains": np.ones(3),
        "sr": 8_000,
        "duration": 1.0,
        "threshold": 0.5,
        "max_events_per_second": 8.0,
        "decay_ms": 120.0,
        "level_db": -12.0,
        "harmonic_limit_hz": 3_000.0,
    }

    unchanged = _add_event_layer(audio.copy(), event_voice="none", **common)
    with_drop = _add_event_layer(audio.copy(), event_voice="water_drop", **common)

    np.testing.assert_array_equal(unchanged, audio)
    assert np.any(with_drop != 0)
    assert np.all(with_drop[:2_000] == 0)
    assert np.any(with_drop[2_000:3_000] != 0)


def test_unknown_event_voice_fails_clearly():
    with pytest.raises(ValueError, match="unknown event voice"):
        _add_event_layer(
            np.zeros(100),
            event_voice="rain",
            salience=np.ones((2, 2)),
            frequencies=np.ones(2) * 100,
            compensation_gains=np.ones(2),
            sr=1_000,
            duration=0.1,
            threshold=0.5,
            max_events_per_second=10,
            decay_ms=50,
            level_db=-12,
            harmonic_limit_hz=400,
        )
