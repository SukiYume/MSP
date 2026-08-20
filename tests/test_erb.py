import inspect

import numpy as np
import pytest

import radiosonify as rs
import radiosonify._perceptual as perceptual_module
from radiosonify._perceptual import (
    _auditory_band_count,
    _hz_to_mel,
    _loudness_compensation_gains,
    _mapping_envelopes,
    _multitone_phases,
    _prepare_values,
    _resample_time_axis,
    _settings_from_mapping,
    _smooth_envelopes,
    _synthesize_prepared,
    _temporal_salience,
    _triangular_filterbank,
    _true_peak,
)
from radiosonify._perceptual_config import PERCEPTUAL_DEFAULTS
from radiosonify._voices import _phase_modulation_coefficients, _render_retro_voice
from radiosonify.erb import erb_frequencies, erb_sonify, mel_frequencies
from tests.helpers import dominant_frequency


def test_erb_frequencies_are_monotonic_and_match_endpoints():
    frequencies = erb_frequencies(8, 200, 4_000)

    assert frequencies[0] == pytest.approx(200)
    assert frequencies[-1] == pytest.approx(4_000)
    assert np.all(np.diff(frequencies) > 0)
    assert np.std(np.diff(frequencies)) > 0  # ERB 间隔不是线性 Hz 间隔。


def test_mel_frequencies_match_htk_spacing_and_expected_defaults():
    frequencies = mel_frequencies(24)

    assert frequencies[0] == pytest.approx(100)
    assert frequencies[-1] == pytest.approx(2_000)
    np.testing.assert_allclose(
        np.diff(_hz_to_mel(frequencies)), np.diff(_hz_to_mel(frequencies))[0]
    )
    np.testing.assert_allclose(
        frequencies[[1, 12, 18, 22]],
        [143.4481929, 809.0759162, 1_372.6293256, 1_860.91603275],
    )


def test_automatic_band_count_uses_the_auditory_erb_span():
    assert _auditory_band_count(100, 2_000) == 18
    settings = _settings_from_mapping({"duration": 1.0, **PERCEPTUAL_DEFAULTS})
    assert settings.n_bands == 18


def test_quadratic_multitone_phases_are_deterministic_and_bounded():
    phases = _multitone_phases(8)

    np.testing.assert_array_equal(phases, _multitone_phases(8))
    assert phases[0] == 0
    assert np.all(np.diff(phases) > 0)
    assert phases[-1] < 2 * np.pi * 8


def test_triangular_filterbank_uses_full_axis_and_preserves_constant_brightness():
    constant = np.full((3, 10), 0.375)
    ramp = np.linspace(0, 1, 10).reshape(1, 10)

    filtered_constant = _triangular_filterbank(constant, 6)
    filtered_ramp = _triangular_filterbank(ramp, 6)

    np.testing.assert_allclose(filtered_constant, 0.375)
    np.testing.assert_allclose(_triangular_filterbank(np.ones((2, 10)), 6), 1.0)
    assert 0 <= filtered_ramp[0, 0] < 0.1
    assert 0.9 < filtered_ramp[0, -1] <= 1
    assert np.all(np.diff(filtered_ramp[0]) > 0)


def test_triangular_filterbank_can_upsample_without_empty_bands():
    filtered = _triangular_filterbank(np.array([[0.0, 1.0]]), 7)

    assert filtered.shape == (1, 7)
    assert np.all(np.isfinite(filtered))
    np.testing.assert_allclose(filtered[0], np.linspace(0, 1, 7))


def test_time_downsampling_uses_every_source_frame_and_preserves_mean():
    data = np.arange(10, dtype=np.float64).reshape(10, 1)

    resampled = _resample_time_axis(data, 6)

    np.testing.assert_allclose(resampled[:, 0], [0.4, 2.0, 3.6, 5.4, 7.0, 8.6])
    assert float(np.mean(resampled)) == pytest.approx(float(np.mean(data)))


def test_erb_sonify_has_strict_duration_and_finite_audio(tmp_path):
    output = tmp_path / "matrix.wav"
    audio, sr = erb_sonify(
        np.arange(48, dtype=float).reshape(8, 6) / 47,
        sr=8_000,
        duration=0.125,
        min_freq=200,
        max_freq=3_000,
        n_bands=5,
        output=output,
    )

    assert sr == 8_000
    assert audio.shape == (1_000,)
    assert audio.dtype == np.float32
    assert np.all(np.isfinite(audio))
    assert _true_peak(audio) <= 10 ** (-1 / 20) + 1e-5
    assert np.sqrt(np.mean(audio**2)) <= 10 ** (-20 / 20) + 1e-6
    assert output.is_file()


def test_feature_position_controls_pitch_without_domain_semantics():
    low = np.zeros((4, 4))
    high = np.zeros((4, 4))
    low[:, 0] = 1
    high[:, -1] = 1

    low_audio, sr = erb_sonify(
        low,
        sr=8_000,
        duration=1,
        min_freq=300,
        max_freq=2_000,
        n_bands=4,
    )
    high_audio, _ = erb_sonify(
        high,
        sr=sr,
        duration=1,
        min_freq=300,
        max_freq=2_000,
        n_bands=4,
    )

    assert dominant_frequency(low_audio, sr) == pytest.approx(300, abs=2)
    assert dominant_frequency(high_audio, sr) == pytest.approx(2_000, abs=2)


def test_descending_frequency_order_reverses_the_feature_mapping():
    data = np.zeros((4, 4))
    data[:, 0] = 1

    ascending, sr = erb_sonify(
        data,
        sr=8_000,
        duration=1,
        min_freq=300,
        max_freq=2_000,
        n_bands=4,
    )
    descending, _ = erb_sonify(
        data,
        sr=sr,
        duration=1,
        min_freq=300,
        max_freq=2_000,
        n_bands=4,
        frequency_order="descending",
    )

    assert dominant_frequency(ascending, sr) < dominant_frequency(descending, sr)


def test_frequency_scale_selects_mel_or_erb_centers():
    data = np.zeros((8, 5))
    data[:, 2] = 0.5

    mel_audio, sr = erb_sonify(
        data,
        sr=8_000,
        duration=1,
        min_freq=100,
        max_freq=2_000,
        n_bands=5,
        frequency_scale="mel",
    )
    erb_audio, _ = erb_sonify(
        data,
        sr=sr,
        duration=1,
        min_freq=100,
        max_freq=2_000,
        n_bands=5,
        frequency_scale="erb",
    )

    assert dominant_frequency(mel_audio, sr) == pytest.approx(
        mel_frequencies(5, 100, 2_000)[2], abs=2
    )
    assert dominant_frequency(erb_audio, sr) == pytest.approx(
        erb_frequencies(5, 100, 2_000)[2], abs=2
    )
    assert dominant_frequency(mel_audio, sr) != pytest.approx(dominant_frequency(erb_audio, sr))


def test_first_axis_is_scanned_as_time():
    data = np.zeros((2, 8))
    data[1, :] = 1
    rows, _ = erb_sonify(data, sr=4_000, duration=0.2, max_freq=1_500, n_bands=2)

    first_half = np.sqrt(np.mean(rows[: len(rows) // 2] ** 2))
    second_half = np.sqrt(np.mean(rows[len(rows) // 2 :] ** 2))
    assert second_half > first_half * 2


def test_declaring_time_axis_on_the_input_gives_identical_audio():
    """轴语义在输入契约里解决，因此转置 + time_axis=1 必须与原布局逐样本一致。

    这条不变量在轴参数还挂在 method_params 上时是**不成立**的：预处理已经沿
    轴 0 扣过基线，方法层才纠正轴顺序就已经晚了，而且不会报错。
    """
    rng = np.random.default_rng(7)
    data = rng.normal(size=(48, 12))

    rows = rs.sonify(data, data_duration=0.3, method="erb", method_params={"n_bands": 4})
    columns = rs.sonify(
        rs.SonificationInput(data.T, duration=0.3, time_axis=1),
        method="erb",
        method_params={"n_bands": 4},
    )

    np.testing.assert_allclose(rows.audio, columns.audio)


def test_power_scale_is_a_mapping_after_shared_normalization():
    data = np.array([0.0, 0.25, 1.0])

    amplitude = _prepare_values(data, value_scale="amplitude", gamma=1)
    power = _prepare_values(data, value_scale="power", gamma=1)

    np.testing.assert_allclose(amplitude, [0, 0.25, 1])
    np.testing.assert_allclose(power, [0, 0.5, 1])


def test_complete_mapping_keeps_ambient_brightness_and_emphasizes_continuous_detail():
    values = np.full((32, 8), 0.2)
    values[16, 3] = 0.8

    envelopes = _mapping_envelopes(
        values,
        salience=_temporal_salience(values),
        value_scale="amplitude",
        gamma=4.0,
        level_db=-6,
        ambient_level_db=-30,
    )
    audio, _ = erb_sonify(values, sr=8_000, duration=0.1, max_freq=2_000, n_bands=8)

    assert np.all(envelopes > 0)
    expected_ambient = 10 ** (-6 / 20) * 10 ** (-30 / 20) * 0.2**4
    assert envelopes[0, 3] == pytest.approx(expected_ambient)
    assert envelopes[16, 3] > envelopes[0, 3] * 100
    assert np.sqrt(np.mean(audio**2)) > 0


def test_mapping_keeps_every_nonzero_band_instead_of_selecting_top_k():
    values = np.full((16, 12), 0.2)
    values[8, :] = np.linspace(0.6, 1.0, 12)

    envelopes = _mapping_envelopes(
        values,
        salience=_temporal_salience(values),
        value_scale="amplitude",
        gamma=4.0,
        level_db=-6,
        ambient_level_db=-30,
    )

    assert np.count_nonzero(envelopes[8]) == 12
    assert np.all(np.diff(envelopes[8]) > 0)


def test_optional_event_salience_is_independent_of_mapping_gamma():
    values = np.full((16, 4), 0.2)
    values[8, :] = [0.5, 0.6, 0.7, 0.8]

    salience = _temporal_salience(values)
    low_gamma = _mapping_envelopes(
        values,
        salience=salience,
        value_scale="amplitude",
        gamma=1.0,
        level_db=-6,
        ambient_level_db=-30,
    )
    high_gamma = _mapping_envelopes(
        values,
        salience=salience,
        value_scale="amplitude",
        gamma=4.0,
        level_db=-6,
        ambient_level_db=-30,
    )

    np.testing.assert_array_equal(salience, _temporal_salience(values))
    assert not np.allclose(low_gamma, high_gamma)
    assert np.count_nonzero(salience[8]) == 4


def test_synthesis_reuses_one_salience_map_for_continuous_and_event_layers(monkeypatch):
    prepared = np.zeros((16, 4))
    prepared[8, 2] = 1.0
    settings = _settings_from_mapping(
        {
            "duration": 0.02,
            **PERCEPTUAL_DEFAULTS,
            "sr": 2_000,
            "min_freq": 100,
            "max_freq": 800,
            "n_bands": 4,
            "event_voice": "water_drop",
        }
    )
    original = perceptual_module._temporal_salience
    calls = 0

    def counted(values):
        nonlocal calls
        calls += 1
        return original(values)

    monkeypatch.setattr(perceptual_module, "_temporal_salience", counted)

    _synthesize_prepared(prepared, settings=settings)

    assert calls == 1


def test_synthesis_is_the_direct_sum_of_independent_band_envelopes(monkeypatch):
    prepared = np.array([[0.1, 0.3], [0.4, 0.2], [0.8, 0.6], [0.2, 0.1]])
    settings = _settings_from_mapping(
        {
            "duration": 0.008,
            **PERCEPTUAL_DEFAULTS,
            "sr": 1_000,
            "min_freq": 100,
            "max_freq": 400,
            "n_bands": 2,
            "gamma": 1,
            "attack_ms": 0,
            "release_ms": 0,
            "loudness_compensation_db": 0,
        }
    )

    def constant_carrier(sample_time, *, band_index, **_kwargs):
        return np.full_like(sample_time, band_index + 1.0)

    monkeypatch.setattr(perceptual_module, "_render_band_voice", constant_carrier)
    audio = _synthesize_prepared(prepared, settings=settings)
    mapped = _mapping_envelopes(
        prepared,
        salience=_temporal_salience(prepared),
        value_scale="amplitude",
        gamma=1,
        level_db=0,
        ambient_level_db=-30,
    )
    sample_envelopes = _resample_time_axis(mapped, audio.size)
    expected = (sample_envelopes[:, 0] + 2 * sample_envelopes[:, 1]) / np.sqrt(2)

    np.testing.assert_allclose(audio, expected)


def test_dense_mel_carriers_do_not_form_a_strong_periodic_amplitude_comb():
    sr = 8_000
    audio, _ = erb_sonify(
        np.full((256, 64), 0.5),
        sr=sr,
        duration=1.0,
        max_freq=2_000,
        n_bands=64,
        gamma=1,
        attack_ms=0,
        release_ms=0,
        loudness_compensation_db=0,
        rms_limit_dbfs=0,
        peak_limit_dbfs=0,
    )
    window = round(sr * 0.012)
    power = np.convolve(audio.astype(np.float64) ** 2, np.ones(window) / window, mode="valid")
    envelope = np.sqrt(power)[:: round(sr / 500)]

    assert np.std(envelope) / np.mean(envelope) < 0.2


def test_attack_release_smoothing_reduces_edges_and_retains_a_tail():
    envelope = np.zeros((20, 1))
    envelope[5:10] = 1

    smoothed = _smooth_envelopes(
        envelope,
        duration=0.2,
        attack_ms=20,
        release_ms=60,
    )

    assert 0 < smoothed[5, 0] < 1
    assert smoothed[10, 0] > 0
    assert smoothed[-1, 0] < smoothed[10, 0]


def test_loudness_compensation_is_bounded_and_favors_low_frequencies():
    frequencies = np.array([100.0, 1_000.0, 2_000.0])
    gains = _loudness_compensation_gains(frequencies, limit_db=6)

    assert gains[0] == pytest.approx(10 ** (6 / 20))
    assert gains[0] > gains[1] > gains[2]
    assert np.all(np.abs(20 * np.log10(gains)) <= 6 + 1e-12)


def test_retro_phase_modulation_keeps_a_dominant_normalized_fundamental():
    coefficients = _phase_modulation_coefficients(0.6, 6)

    assert np.linalg.norm(coefficients) == pytest.approx(1)
    assert coefficients[0] > 0
    assert abs(coefficients[0]) > np.max(np.abs(coefficients[1:]))
    assert np.all(np.diff(np.abs(coefficients)) < 0)


def test_retro_voice_adds_harmonics_inside_the_requested_frequency_limit():
    sr = 48_000
    sample_time = np.arange(sr, dtype=np.float64) / sr
    voice = _render_retro_voice(
        sample_time,
        frequency=900,
        phase=0.2,
        band_index=0,
        harmonic_limit_hz=2_500,
        detune_cents=0,
        fm_index=0.6,
        chorus_rate_hz=0,
        chorus_depth_ms=0,
    )
    spectrum = np.abs(np.fft.rfft(voice))
    frequencies = np.fft.rfftfreq(voice.size, 1 / sr)
    fundamental = spectrum[np.argmin(np.abs(frequencies - 900))]
    second_harmonic = spectrum[np.argmin(np.abs(frequencies - 1_800))]

    assert frequencies[np.argmax(spectrum[1:]) + 1] == pytest.approx(900, abs=1)
    assert second_harmonic > fundamental * 0.1
    assert np.sum(spectrum[frequencies > 2_500] ** 2) / np.sum(spectrum**2) < 1e-20


def test_timbre_changes_only_the_carrier_and_remains_deterministic():
    background = np.full((64, 8), 0.2)
    event = background.copy()
    event[28:34, 3:5] = 1.0
    settings = {
        "sr": 8_000,
        "duration": 0.25,
        "max_freq": 2_000,
        "n_bands": 8,
    }

    retro_background, _ = erb_sonify(background, timbre="retro_digital", **settings)
    sine_background, _ = erb_sonify(background, timbre="sine", **settings)
    retro_event, _ = erb_sonify(event, timbre="retro_digital", **settings)
    repeated_retro, _ = erb_sonify(event, timbre="retro_digital", **settings)
    sine_event, _ = erb_sonify(event, timbre="sine", **settings)

    np.testing.assert_array_equal(retro_event, repeated_retro)
    assert not np.allclose(retro_background, sine_background)
    assert not np.allclose(retro_event, sine_event)


def test_instrument_palette_and_water_drop_events_are_deterministic():
    data = np.full((64, 8), 0.2)
    data[28:34, 3:5] = 1.0
    settings = {
        "sr": 8_000,
        "duration": 0.5,
        "max_freq": 2_000,
        "n_bands": 8,
        "timbre": "instrument_palette",
    }

    palette, _ = erb_sonify(data, event_voice="none", **settings)
    drops, _ = erb_sonify(
        data,
        event_voice="water_drop",
        event_params={"salience_threshold": 0.1, "max_events_per_second": 8},
        **settings,
    )
    repeated, _ = erb_sonify(
        data,
        event_voice="water_drop",
        event_params={"salience_threshold": 0.1, "max_events_per_second": 8},
        **settings,
    )

    np.testing.assert_array_equal(drops, repeated)
    assert not np.allclose(palette, drops)


def test_constant_minmax_matrix_is_silence():
    audio, _ = erb_sonify(
        rs.preprocess(np.ones((4, 4))),
        sr=4_000,
        duration=0.1,
        max_freq=1_500,
    )

    assert np.all(audio == 0)


def test_public_erb_parameters_group_voice_and_event_details():
    parameters = inspect.signature(erb_sonify).parameters

    assert "voice_params" in parameters
    assert "event_params" in parameters
    assert "mapping_level_db" in parameters
    assert "ambient_level_db" in parameters
    for removed in (
        "background_level_db",
        "foreground_threshold",
        "max_polyphony",
        "detune_cents",
        "event_decay_ms",
    ):
        assert removed not in parameters


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_freq": 4_000, "sr": 8_000}, "Nyquist"),
        ({"min_freq": 1_000, "max_freq": 500}, "max_freq"),
        ({"n_bands": 0}, "n_bands"),
        ({"value_scale": "db"}, "value_scale"),
        ({"gamma": 0}, "gamma"),
        ({"frequency_scale": "bark"}, "frequency_scale"),
        ({"timbre": "brass"}, "timbre"),
        ({"mapping_level_db": 1}, "mapping_level_db"),
        ({"ambient_level_db": 1}, "ambient_level_db"),
        ({"voice_params": {"harmonic_limit_hz": 0}}, "harmonic_limit_hz"),
        ({"voice_params": {"detune_cents": -1}}, "detune_cents"),
        ({"voice_params": {"detune_cents": 51}}, "detune_cents"),
        ({"voice_params": {"fm_index": -0.1}}, "fm_index"),
        ({"voice_params": {"fm_index": 1.1}}, "fm_index"),
        ({"voice_params": {"chorus_rate_hz": -0.1}}, "chorus_rate_hz"),
        ({"voice_params": {"chorus_rate_hz": 10.1}}, "chorus_rate_hz"),
        ({"voice_params": {"chorus_depth_ms": -0.1}}, "chorus_depth_ms"),
        ({"voice_params": {"chorus_depth_ms": 20.1}}, "chorus_depth_ms"),
        ({"voice_params": {"unknown": 1}}, "unknown voice_params"),
        ({"voice_params": 1}, "voice_params"),
        ({"event_voice": "rain"}, "event_voice"),
        ({"event_params": {"salience_threshold": -0.1}}, "salience_threshold"),
        ({"event_params": {"salience_threshold": 1.1}}, "salience_threshold"),
        ({"event_params": {"max_events_per_second": -1}}, "max_events_per_second"),
        ({"event_params": {"max_events_per_second": 101}}, "max_events_per_second"),
        ({"event_params": {"decay_ms": 0}}, "decay_ms"),
        ({"event_params": {"decay_ms": 5_001}}, "decay_ms"),
        ({"event_params": {"level_db": 1}}, "level_db"),
        ({"event_params": {"unknown": 1}}, "unknown event_params"),
        ({"event_params": 1}, "event_params"),
        ({"attack_ms": -1}, "attack_ms"),
        ({"release_ms": -1}, "release_ms"),
        ({"loudness_compensation_db": -1}, "loudness_compensation_db"),
        ({"rms_limit_dbfs": 1}, "rms_limit_dbfs"),
        ({"peak_limit_dbfs": 1}, "peak_limit_dbfs"),
    ],
)
def test_invalid_erb_parameters_fail_clearly(kwargs, message):
    with pytest.raises(ValueError, match=message):
        erb_sonify(np.ones((4, 4)), duration=0.01, **kwargs)


def test_low_level_erb_rejects_data_that_skipped_shared_preprocessing():
    with pytest.raises(ValueError, match="preprocess"):
        erb_sonify(np.arange(16.0).reshape(4, 4), duration=0.01)


def test_matrix_auto_method_is_erb_and_image_alias_is_accepted():
    result = rs.sonify(
        np.eye(4),
        data_duration=0.02,
        data_type="image",
        method_params={"sr": 8_000, "max_freq": 3_000, "n_bands": 4},
    )

    assert result.data_type is rs.DataType.MATRIX
    assert result.method == "erb"
    assert result.repeat == 1
    assert result.audio.shape == (160,)
