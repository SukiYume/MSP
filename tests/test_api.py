import importlib

import numpy as np
import pytest
import soundfile as sf

import radiosonify as rs
import radiosonify.api as api_module

amplitude_module = importlib.import_module("radiosonify.amplitude")
griffinlim_module = importlib.import_module("radiosonify.griffinlim")
hifigan_module = importlib.import_module("radiosonify.hifigan")
musicnet_module = importlib.import_module("radiosonify.musicnet")


def _dominant_frequency(audio, sr):
    frequencies = np.fft.rfftfreq(len(audio), d=1 / sr)
    return frequencies[np.argmax(np.abs(np.fft.rfft(audio)))]


def test_unified_profile_auto_method_uses_physical_duration_and_speed(tmp_path):
    output = tmp_path / "nested" / "profile.wav"
    result = rs.sonify(
        np.linspace(0, 1, 32),
        data_duration=0.4,
        method="auto",
        speed=2,
        method_params={"sr": 8_000, "freq": 440},
        output=output,
    )

    assert result.data_type is rs.DataType.PROFILE
    assert result.method == "amplitude"
    assert result.target_duration == pytest.approx(0.2)
    assert result.output_duration == pytest.approx(0.2)
    assert len(result.audio) == 1_600
    assert result.method_params["freq"] == 440
    assert result.output_path == output
    assert result.audio[0] == pytest.approx(0)
    assert result.audio[-1] == pytest.approx(0)
    wav, wav_sr = sf.read(output)
    assert len(wav) == 1_600
    assert wav_sr == 8_000


def test_unified_dynamic_auto_method_is_duration_fitted(monkeypatch):
    calls = []

    def fake_griffinlim(data, output, **params):
        calls.append((data.shape, output, params))
        return np.sin(np.linspace(0, 10, 777)), 1_000

    monkeypatch.setattr(griffinlim_module, "griffinlim", fake_griffinlim)
    source = rs.SonificationInput(
        np.ones((32, 16)),
        duration=2,
        data_type="dynamic_spectrum",
        name="real-event",
    )

    result = rs.sonify(
        source,
        speed=4,
        method_params={"n_iter": 3, "freq_rebin": 16, "n_fft": 64},
    )

    assert result.method == "griffinlim"
    assert result.source_name == "real-event"
    assert len(result.audio) == 500
    assert result.output_duration == pytest.approx(0.5)
    assert result.method_sample_rate == 1_000
    assert result.method_native_samples == 777
    assert result.method_native_duration == pytest.approx(0.777)
    assert result.method_time_scale == pytest.approx(500 / 777)
    assert calls[0][0] == (32, 16)
    assert calls[0][2]["n_iter"] == 3


def test_musicnet_is_an_optional_audio_postprocessor(monkeypatch):
    def fake_musicnet(input_audio, sr, output, **params):
        assert len(input_audio) == 2_000
        assert output is None
        assert params["decoder_id"] == 4
        return np.asarray(input_audio)[::2], sr

    monkeypatch.setattr(musicnet_module, "musicnet", fake_musicnet)
    result = rs.sonify(
        np.linspace(0, 1, 16),
        data_duration=0.25,
        method_params={"sr": 8_000},
        postprocess="musicnet",
        postprocess_params={"decoder_id": 4},
    )

    assert result.postprocess == "musicnet"
    assert result.postprocess_params["decoder_id"] == 4
    assert "batch_size" not in result.postprocess_params
    assert result.postprocess_native_samples == 1_000
    assert result.postprocess_native_duration == pytest.approx(0.125)
    assert result.postprocess_time_scale == pytest.approx(2)
    assert len(result.audio) == 2_000


def test_musicnet_runs_before_playback_speed_expands_duration(monkeypatch):
    calls = []

    def fake_musicnet(input_audio, sr, output, **params):
        calls.append((len(input_audio), sr, output, params))
        return np.asarray(input_audio), sr

    monkeypatch.setattr(musicnet_module, "musicnet", fake_musicnet)
    result = rs.sonify(
        np.linspace(0, 1, 16),
        data_duration=0.25,
        speed=0.1,
        method_params={"sr": 8_000},
        postprocess="musicnet",
    )

    assert calls[0][0:3] == (2_000, 8_000, None)
    assert result.target_duration == pytest.approx(2.5)
    assert len(result.audio) == 20_000


def test_array_requires_data_duration():
    with pytest.raises(ValueError, match="data_duration is required"):
        rs.sonify(np.ones(16))


def test_existing_input_rejects_duplicate_metadata():
    source = rs.SonificationInput(np.ones(16), duration=1)
    with pytest.raises(ValueError, match="stored in SonificationInput"):
        rs.sonify(source, data_duration=1)


def test_method_type_and_parameter_errors_are_actionable():
    with pytest.raises(ValueError, match="does not accept profile"):
        rs.sonify(np.ones(16), data_duration=1, method="hifigan")
    with pytest.raises(ValueError, match=r"unknown parameter\(s\).*carrier"):
        rs.sonify(
            np.ones(16),
            data_duration=1,
            method="amplitude",
            method_params={"carrier": 440},
        )


def test_unified_griffinlim_translates_deprecated_n_mels(monkeypatch):
    calls = []

    def fake_griffinlim(data, output, **params):
        calls.append(params)
        return np.linspace(-1, 1, 1_000), 1_000

    monkeypatch.setattr(griffinlim_module, "griffinlim", fake_griffinlim)
    with pytest.warns(DeprecationWarning, match="freq_rebin"):
        rs.sonify(
            np.ones((8, 8)),
            data_duration=1,
            method="griffinlim",
            method_params={"n_mels": 8},
        )

    assert calls[0]["freq_rebin"] == 8
    assert "n_mels" not in calls[0]

    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="cannot both be supplied"):
            rs.sonify(
                np.ones((8, 8)),
                data_duration=1,
                method="griffinlim",
                method_params={"n_mels": 8, "freq_rebin": 8},
            )


def test_profile_method_unified_defaults_do_not_download_instrument():
    result = rs.sonify(
        np.linspace(0, 1, 16),
        data_duration=0.01,
        method="profile",
        method_params={"sr": 8_000},
    )

    assert result.repeat == 1
    assert "repeat" not in result.method_params
    assert result.method_params["instrument"] is None
    assert len(result.audio) == 80


@pytest.mark.parametrize("method", ["profile", "amplitude"])
def test_unified_repeat_expands_duration_and_is_recorded(method):
    result = rs.sonify(
        np.linspace(0, 1, 16),
        data_duration=0.01,
        method=method,
        repeat=5,
        method_params={"sr": 8_000},
    )

    assert result.repeat == 5
    assert result.target_duration == pytest.approx(0.05)
    assert result.output_duration == pytest.approx(0.05)
    assert len(result.audio) == 400


_DURATION_CONTRACT_CASES = [
    pytest.param(method, speed, repeat, id=f"{method}-speed-{speed}-repeat-{repeat}")
    for method, repeats in (
        ("profile", (1, 3)),
        ("amplitude", (1, 3)),
        ("griffinlim", (1,)),
        ("hifigan", (1,)),
    )
    for speed in (0.1, 1.0, 2.0)
    for repeat in repeats
]


@pytest.mark.parametrize("method,speed,repeat", _DURATION_CONTRACT_CASES)
def test_cross_method_output_duration_contract(method, speed, repeat, monkeypatch):
    def fake_dynamic_method(data, output, **params):
        del data, output, params
        return np.linspace(-1, 1, 137), 1_000

    monkeypatch.setattr(griffinlim_module, "griffinlim", fake_dynamic_method)
    monkeypatch.setattr(hifigan_module, "hifigan", fake_dynamic_method)

    dynamic = method in {"griffinlim", "hifigan"}
    data = np.ones((8, 8)) if dynamic else np.linspace(0, 1, 8)
    method_params = {} if method == "hifigan" else {"sr": 1_000}
    if method == "amplitude":
        method_params["freq"] = 100
    result = rs.sonify(
        data,
        data_duration=0.02,
        method=method,
        speed=speed,
        repeat=repeat,
        method_params=method_params,
    )

    expected_duration = 0.02 * repeat / speed
    assert result.target_duration == pytest.approx(expected_duration)
    assert result.output_duration == pytest.approx(expected_duration)
    assert len(result.audio) == round(result.sample_rate * expected_duration)


def test_output_sr_normalizes_all_primary_method_containers(monkeypatch):
    def fake_griffinlim(data, output, **params):
        del data, output, params
        return np.linspace(-1, 1, 320), 16_000

    def fake_hifigan(data, output, **params):
        del data, output, params
        return np.linspace(-1, 1, 441), 22_050

    monkeypatch.setattr(griffinlim_module, "griffinlim", fake_griffinlim)
    monkeypatch.setattr(hifigan_module, "hifigan", fake_hifigan)
    cases = (
        ("profile", np.linspace(0, 1, 8), {"sr": 8_000}, 8_000),
        ("amplitude", np.linspace(0, 1, 8), {"sr": 12_000}, 12_000),
        ("griffinlim", np.ones((8, 8)), {}, 16_000),
        ("hifigan", np.ones((8, 8)), {}, 22_050),
    )

    for method, data, method_params, native_sr in cases:
        result = rs.sonify(
            data,
            data_duration=0.02,
            method=method,
            method_params=method_params,
            output_sr=48_000,
        )

        assert result.method == method
        assert result.method_sample_rate == native_sr
        assert result.sample_rate == 48_000
        assert result.output_duration == pytest.approx(0.02)
        assert len(result.audio) == 960


def test_output_sr_normalizes_musicnet_postprocessing(monkeypatch):
    def fake_musicnet(input_audio, sr, output, **params):
        del input_audio, sr, output, params
        return np.linspace(-1, 1, 4_000), 16_000

    monkeypatch.setattr(musicnet_module, "musicnet", fake_musicnet)
    result = rs.sonify(
        np.linspace(0, 1, 16),
        data_duration=0.25,
        method_params={"sr": 8_000},
        postprocess="musicnet",
        output_sr=48_000,
    )

    assert result.method_sample_rate == 8_000
    assert result.postprocess_native_samples == 4_000
    assert result.postprocess_native_duration == pytest.approx(0.25)
    assert result.sample_rate == 48_000
    assert len(result.audio) == 12_000
    assert result.output_duration == pytest.approx(0.25)


def test_output_sr_conversion_preserves_physical_pitch_even_when_requested(monkeypatch):
    native_sr = 16_000
    duration = 0.25
    tone = np.sin(2 * np.pi * 440 * np.arange(round(native_sr * duration)) / native_sr)

    def fake_hifigan(data, output, **params):
        del data, output, params
        return tone, native_sr

    monkeypatch.setattr(hifigan_module, "hifigan", fake_hifigan)
    result = rs.sonify(
        np.ones((8, 8)),
        data_duration=duration,
        method="hifigan",
        preserve_pitch=True,
        output_sr=48_000,
    )

    assert result.sample_rate == 48_000
    assert len(result.audio) == 12_000
    assert _dominant_frequency(result.audio, result.sample_rate) == pytest.approx(440, abs=5)


@pytest.mark.parametrize("output_sr", [0, -1, 1.5, True, "48000", 10**400])
def test_invalid_output_sr_fails_before_synthesis(output_sr, monkeypatch):
    monkeypatch.setattr(
        amplitude_module,
        "amplitude_modulate",
        lambda *args, **kwargs: pytest.fail("synthesis should not run"),
    )

    with pytest.raises(ValueError, match="output_sr"):
        rs.sonify(
            np.ones(16),
            data_duration=1,
            output_sr=output_sr,
        )


def test_repeat_rejects_invalid_values_and_neural_methods(monkeypatch):
    with pytest.raises(ValueError, match="repeat"):
        rs.sonify(np.arange(8.0), data_duration=1, repeat=0)

    monkeypatch.setattr(
        hifigan_module,
        "hifigan",
        lambda *args, **kwargs: pytest.fail("synthesis should not run"),
    )
    with pytest.raises(ValueError, match="only supported"):
        rs.sonify(
            np.ones((4, 4)),
            data_duration=1,
            method="hifigan",
            repeat=2,
        )


def test_repeat_is_a_control_parameter_not_a_method_parameter():
    with pytest.raises(ValueError, match=r"unknown parameter.*repeat"):
        rs.sonify(
            np.arange(8.0),
            data_duration=1,
            method="profile",
            method_params={"repeat": 5},
        )


def test_postprocess_parameters_require_a_postprocessor():
    with pytest.raises(ValueError, match="requires postprocess"):
        rs.sonify(
            np.ones(16),
            data_duration=1,
            postprocess_params={"decoder_id": 2},
        )


def test_removed_musicnet_batch_size_is_rejected_before_synthesis(monkeypatch):
    monkeypatch.setattr(
        amplitude_module,
        "amplitude_modulate",
        lambda *args, **kwargs: pytest.fail("synthesis should not run"),
    )

    with pytest.raises(ValueError, match=r"unknown MusicNet.*batch_size"):
        rs.sonify(
            np.ones(16),
            data_duration=1,
            postprocess="musicnet",
            postprocess_params={"batch_size": 8},
        )


def test_control_and_postprocess_validation_happen_before_synthesis(monkeypatch):
    def should_not_run(*args, **kwargs):
        raise AssertionError("primary synthesis should not run")

    monkeypatch.setattr(amplitude_module, "amplitude_modulate", should_not_run)

    with pytest.raises(ValueError, match="preserve_pitch"):
        rs.sonify(
            np.ones(16),
            data_duration=1,
            preserve_pitch="yes",
        )
    with pytest.raises(ValueError, match="unknown MusicNet"):
        rs.sonify(
            np.ones(16),
            data_duration=1,
            postprocess="musicnet",
            postprocess_params={"style": 2},
        )
    with pytest.raises(ValueError, match="decoder_id"):
        rs.sonify(
            np.ones(16),
            data_duration=1,
            postprocess="musicnet",
            postprocess_params={"decoder_id": 9},
        )


def test_parameter_mapping_keys_must_be_strings():
    with pytest.raises(ValueError, match="keys must be strings"):
        rs.sonify(
            np.ones(16),
            data_duration=1,
            method_params={1: 440},
        )
    with pytest.raises(ValueError, match="keys must be strings"):
        rs.sonify(
            np.ones(16),
            data_duration=1,
            postprocess="musicnet",
            postprocess_params={1: 2},
        )


def test_result_parameter_provenance_is_read_only():
    result = rs.sonify(
        np.linspace(0, 1, 16),
        data_duration=0.01,
        method_params={"sr": 8_000},
    )

    with pytest.raises(TypeError):
        result.method_params["freq"] = 123
    with pytest.raises(ValueError, match="read-only"):
        result.audio[0] = 123


def test_result_equality_and_hashing_use_identity_without_array_errors():
    kwargs = {
        "data_duration": 0.01,
        "method_params": {"sr": 8_000},
    }
    first = rs.sonify(np.linspace(0, 1, 16), **kwargs)
    second = rs.sonify(np.linspace(0, 1, 16), **kwargs)

    assert first == first
    assert first != second
    assert {first: "first", second: "second"}[first] == "first"


def test_without_postprocess_duration_is_fitted_once(monkeypatch):
    original = api_module.fit_audio_duration
    calls = []

    def counted_fit(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(api_module, "fit_audio_duration", counted_fit)
    rs.sonify(
        np.linspace(0, 1, 16),
        data_duration=0.01,
        method_params={"sr": 8_000},
    )

    assert len(calls) == 1


def test_invalid_output_path_fails_before_synthesis(monkeypatch, tmp_path):
    def should_not_run(*args, **kwargs):
        raise AssertionError("synthesis should not run")

    monkeypatch.setattr(amplitude_module, "amplitude_modulate", should_not_run)
    with pytest.raises(ValueError, match=r"\.wav"):
        rs.sonify(
            np.ones(16),
            data_duration=1,
            output=tmp_path / "bad.flac",
        )
