import warnings

import numpy as np
import pytest

from radiosonify.timing import (
    _resample_audio_rate,
    condition_audio_output,
    duration_to_frames,
    duration_to_samples,
    fit_audio_duration,
    target_audio_duration,
)
from tests.helpers import dominant_frequency


def test_speed_semantics_and_sample_rounding():
    assert target_audio_duration(10, speed=2) == pytest.approx(5)
    assert target_audio_duration(10, speed=0.5) == pytest.approx(20)
    assert target_audio_duration(10, speed=2, repeat=5) == pytest.approx(25)
    assert duration_to_samples(1 / 3, 48_000) == 16_000
    assert duration_to_frames(3.0, 22_050, 256) == 258
    assert duration_to_frames(128 * 256 / 22_050, 22_050, 256) == 128


@pytest.mark.parametrize("duration", [0.5, 1.75])
def test_playback_resampling_produces_exact_sample_count(duration):
    sr = 8_000
    audio = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)

    transformed = fit_audio_duration(audio, sr, duration)

    assert len(transformed) == round(sr * duration)
    assert transformed.dtype == np.float32
    assert np.all(np.isfinite(transformed))
    assert np.max(np.abs(transformed)) <= 1


def test_stereo_resampling_preserves_channels_and_exact_sample_count():
    sr = 8_000
    time = np.arange(sr) / sr
    stereo = np.column_stack((np.sin(2 * np.pi * 440 * time), np.sin(2 * np.pi * 880 * time)))

    transformed = fit_audio_duration(stereo, sr, 0.5)

    assert transformed.shape == (4_000, 2)
    assert dominant_frequency(transformed[:, 0], sr) == pytest.approx(880, abs=5)
    assert dominant_frequency(transformed[:, 1], sr) == pytest.approx(1_760, abs=5)


def test_stereo_pitch_preserving_stretch_keeps_both_channel_pitches():
    sr = 8_000
    time = np.arange(sr) / sr
    stereo = np.column_stack((np.sin(2 * np.pi * 440 * time), np.sin(2 * np.pi * 660 * time)))

    transformed = fit_audio_duration(stereo, sr, 0.5, preserve_pitch=True)

    assert transformed.shape == (4_000, 2)
    assert dominant_frequency(transformed[:, 0], sr) == pytest.approx(440, abs=15)
    assert dominant_frequency(transformed[:, 1], sr) == pytest.approx(660, abs=15)


def test_default_speed_change_moves_pitch_but_preserve_pitch_does_not():
    sr = 8_000
    audio = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)

    playback_speed = fit_audio_duration(audio, sr, 0.5)
    pitch_preserved = fit_audio_duration(audio, sr, 0.5, preserve_pitch=True)

    assert dominant_frequency(playback_speed, sr) == pytest.approx(880, abs=5)
    assert dominant_frequency(pitch_preserved, sr) == pytest.approx(440, abs=15)
    assert len(playback_speed) == len(pitch_preserved) == 4_000


def test_pitch_preserving_short_event_uses_an_adaptive_stft_window():
    audio = np.sin(2 * np.pi * 100 * np.arange(1_000) / 4_000)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transformed = fit_audio_duration(
            audio,
            4_000,
            0.125,
            preserve_pitch=True,
        )

    assert len(transformed) == 500
    assert not any("n_fft" in str(item.message) for item in caught)


def test_silence_stays_silent_when_length_changes():
    transformed = fit_audio_duration(np.zeros(100), 1_000, 0.25)

    assert len(transformed) == 250
    assert np.all(transformed == 0)


def test_extreme_downsampling_ratios_remain_positive_and_exact():
    audio = np.linspace(-1, 1, 25_000)

    by_rate = _resample_audio_rate(audio, 25_000, 1)
    by_duration = fit_audio_duration(audio, 1, 1.0)

    assert by_rate.shape == by_duration.shape == (1,)
    assert np.all(np.isfinite(by_rate))
    assert np.all(np.isfinite(by_duration))


@pytest.mark.parametrize("readonly", [False, True])
def test_fit_audio_duration_does_not_modify_equal_length_input(readonly):
    audio = np.array([0.0, 2.0, -2.0, 1.0], dtype=np.float64)
    original = audio.copy()
    audio.setflags(write=not readonly)

    transformed = fit_audio_duration(audio, 4, 1.0)

    np.testing.assert_array_equal(audio, original)
    np.testing.assert_allclose(transformed, [0.0, 1.0, -1.0, 0.5])


def test_output_conditioning_removes_dc_tapers_edges_and_normalizes_peak():
    sr = 8_000
    audio = 0.4 + 0.2 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)

    conditioned = condition_audio_output(audio, sr)

    assert len(conditioned) == len(audio)
    assert conditioned.dtype == np.float32
    assert abs(float(np.mean(conditioned))) < 1e-7
    assert conditioned[0] == pytest.approx(0)
    assert conditioned[-1] == pytest.approx(0)
    assert np.max(np.abs(conditioned)) == pytest.approx(0.9)


def test_stereo_output_conditioning_handles_each_channel_dc_without_downmixing():
    sr = 8_000
    time = np.arange(sr) / sr
    stereo = np.column_stack(
        (0.4 + 0.2 * np.sin(2 * np.pi * 440 * time), -0.3 + 0.1 * np.sin(2 * np.pi * 880 * time))
    )

    conditioned = condition_audio_output(stereo, sr)

    assert conditioned.shape == stereo.shape
    np.testing.assert_allclose(np.mean(conditioned, axis=0), [0, 0], atol=1e-7)
    np.testing.assert_allclose(conditioned[[0, -1]], 0)
    assert np.max(np.abs(conditioned)) == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("audio", "expected"),
    [
        ([2.0], [0.9]),
        ([1.0, -0.5], [0.9, -0.45]),
    ],
)
def test_output_conditioning_preserves_one_or_two_sample_signals(audio, expected):
    conditioned = condition_audio_output(np.asarray(audio), 100)

    np.testing.assert_allclose(conditioned, expected)


def test_output_conditioning_handles_sample_rates_with_subsample_fade_duration():
    audio = np.sin(np.linspace(0, 4 * np.pi, 100))

    conditioned = condition_audio_output(audio, 100)

    assert len(conditioned) == 100
    assert np.all(np.isfinite(conditioned))
    assert np.max(np.abs(conditioned)) == pytest.approx(0.9)


def test_output_conditioning_can_preserve_a_quiet_input_peak():
    sr = 8_000
    audio = 0.02 + 0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
    input_peak = float(np.max(np.abs(audio)))

    conditioned = condition_audio_output(audio, sr, peak=None)

    assert np.max(np.abs(conditioned)) == pytest.approx(input_peak)
    assert np.max(np.abs(conditioned)) < 0.2
    assert abs(float(np.mean(conditioned))) < 1e-7


def test_output_conditioning_peak_none_only_scales_down_clipping_short_audio():
    quiet = condition_audio_output(np.array([0.1, -0.05]), 100, peak=None)
    loud = condition_audio_output(np.array([2.0, -1.0]), 100, peak=None)

    np.testing.assert_allclose(quiet, [0.1, -0.05])
    np.testing.assert_allclose(loud, [1.0, -0.5])


@pytest.mark.parametrize(
    ("function", "args", "message"),
    [
        (target_audio_duration, (1, 0), "speed"),
        (target_audio_duration, (0, 1), "data_duration"),
        (target_audio_duration, (1, 1, 0), "repeat"),
        (target_audio_duration, (1e308, 1e-308), "finite target"),
        (target_audio_duration, (1, 1, 10**400), "finite target"),
        (duration_to_samples, (0.00001, 1_000), "shorter than one sample"),
        (duration_to_samples, (1e308, 48_000), "too many"),
        (duration_to_samples, (1, 10**400), "too many"),
        (duration_to_frames, (0.00001, 1_000, 256), "shorter than one"),
        (duration_to_frames, (1, 1_000, 0), "hop_length"),
        (fit_audio_duration, (np.ones(10), 1_000, 0.1), "preserve_pitch"),
    ],
)
def test_timing_rejects_invalid_parameters(function, args, message):
    kwargs = {"preserve_pitch": "yes"} if function is fit_audio_duration else {}
    with pytest.raises(ValueError, match=message):
        function(*args, **kwargs)
