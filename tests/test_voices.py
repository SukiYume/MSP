import numpy as np
import pytest

from radiosonify._perceptual_config import PERCEPTUAL_CHOICES
from radiosonify._voices import (
    _palette_components,
    _render_voice,
)


def _voice(name, *, position=0.5):
    sr = 16_000
    sample_time = np.arange(sr, dtype=np.float64) / sr
    return _render_voice(
        name,
        sample_time,
        frequency=300.0,
        phase=0.2,
        band_index=3,
        band_position=position,
        harmonic_limit_hz=2_500.0,
        detune_cents=10.0,
        fm_index=0.6,
        chorus_rate_hz=0.45,
        chorus_depth_ms=8.0,
    )


@pytest.mark.parametrize("position", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_palette_crossfade_is_continuous_and_preserves_common_gain(position):
    components = _palette_components(position)

    assert sum(weight for _, weight in components) == pytest.approx(1.0)
    assert all(weight > 0 for _, weight in components)


def test_palette_uses_expected_anchor_voices():
    np.testing.assert_allclose(_voice("instrument_palette", position=0.0), _voice("warm_pad"))
    np.testing.assert_allclose(
        _voice("instrument_palette", position=0.5),
        _voice("soft_marimba"),
    )
    np.testing.assert_allclose(
        _voice("instrument_palette", position=1.0),
        _voice("glass_bell"),
    )


@pytest.mark.parametrize("name", PERCEPTUAL_CHOICES["timbre"])
def test_procedural_voices_are_finite_deterministic_and_audible(name):
    first = _voice(name)
    second = _voice(name)

    np.testing.assert_array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert 0.2 < np.sqrt(np.mean(first**2)) < 1.0


@pytest.mark.parametrize(
    "name",
    ["warm_pad", "soft_marimba", "glass_bell", "instrument_palette"],
)
def test_new_voices_obey_the_harmonic_ceiling(name):
    voice = _voice(name)
    spectrum = np.abs(np.fft.rfft(voice)) ** 2
    frequencies = np.fft.rfftfreq(voice.size, 1 / 16_000)

    assert spectrum[frequencies > 2_550].sum() / spectrum.sum() < 2e-4


def test_unknown_voice_fails_clearly():
    with pytest.raises(ValueError, match="unknown voice"):
        _voice("brass")
