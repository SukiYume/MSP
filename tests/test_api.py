import importlib
from collections import UserList

import numpy as np
import pytest
import soundfile as sf

import radiosonify as rs
from radiosonify.validation import _freeze_value
from tests.helpers import dominant_frequency

amplitude_module = importlib.import_module("radiosonify.amplitude")
erb_module = importlib.import_module("radiosonify.erb")
griffinlim_module = importlib.import_module("radiosonify.griffinlim")
hifigan_module = importlib.import_module("radiosonify.hifigan")
musicnet_module = importlib.import_module("radiosonify.musicnet")
pipeline_module = importlib.import_module("radiosonify.pipeline")


def _patch_hifigan_runner(monkeypatch, runner):
    """Replace the optional HiFi-GAN runtime as one isolated test double."""
    monkeypatch.setattr(hifigan_module, "hifigan", runner)
    monkeypatch.setattr(hifigan_module, "_preflight_hifigan", lambda: None)


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
    assert result.repeat == 5
    assert result.target_duration == pytest.approx(1.0)
    assert result.output_duration == pytest.approx(1.0)
    assert len(result.audio) == 8_000
    assert result.method_params["compression"] == 0
    assert result.method_params["harmonics"] == 4
    assert result.method_params["freq"] == 440
    assert result.output_path == output
    assert result.audio[0] == pytest.approx(0)
    assert result.audio[-1] == pytest.approx(0)
    wav, wav_sr = sf.read(output)
    assert len(wav) == 8_000
    assert wav_sr == 8_000


def test_unified_matrix_auto_method_is_duration_fitted(monkeypatch):
    calls = []

    def fake_erb(data, output, **params):
        calls.append((data.shape, output, params))
        return np.sin(np.linspace(0, 10, 777)), 1_000

    monkeypatch.setattr(erb_module, "erb_sonify", fake_erb)
    source = rs.SonificationInput(
        np.ones((32, 16)),
        duration=2,
        data_type="dynamic_spectrum",
        name="real-event",
    )

    result = rs.sonify(
        source,
        speed=4,
        method_params={"n_bands": 16},
    )

    assert result.method == "erb"
    assert result.source_name == "real-event"
    assert len(result.audio) == 500
    assert result.output_duration == pytest.approx(0.5)
    assert result.method_sample_rate == 1_000
    assert result.method_native_samples == 777
    assert result.method_native_duration == pytest.approx(0.777)
    assert result.method_time_scale == pytest.approx(500 / 777)
    assert calls[0][0] == (32, 16)
    assert calls[0][2]["n_bands"] == 16
    assert calls[0][2]["duration"] == pytest.approx(0.5)


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
        repeat=1,
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
        repeat=1,
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


@pytest.mark.parametrize(
    ("method", "method_params", "removed_name"),
    [
        ("griffinlim", {"n_mels": 8}, "n_mels"),
        ("griffinlim", {"freq_rebin": 8}, "freq_rebin"),
        ("hifigan", {"time_rebin": 8}, "time_rebin"),
        ("hifigan", {"time_smoothing": 1.0}, "time_smoothing"),
        ("erb", {"time_axis": 1}, "time_axis"),
        ("amplitude", {"time_downsample": 2}, "time_downsample"),
    ],
)
def test_removed_method_level_data_controls_are_unknown(method, method_params, removed_name):
    data = np.ones(8) if method == "amplitude" else np.ones((8, 8))

    with pytest.raises(ValueError, match=rf"unknown parameter.*{removed_name}"):
        rs.sonify(data, data_duration=1, method=method, method_params=method_params)


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
    _patch_hifigan_runner(monkeypatch, fake_dynamic_method)

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
    _patch_hifigan_runner(monkeypatch, fake_hifigan)
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
            repeat=1,
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
        repeat=1,
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

    _patch_hifigan_runner(monkeypatch, fake_hifigan)
    result = rs.sonify(
        np.ones((8, 8)),
        data_duration=duration,
        method="hifigan",
        preserve_pitch=True,
        output_sr=48_000,
    )

    assert result.sample_rate == 48_000
    assert len(result.audio) == 12_000
    assert dominant_frequency(result.audio, result.sample_rate) == pytest.approx(440, abs=5)


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


def test_repeat_rejects_invalid_values():
    with pytest.raises(ValueError, match="repeat"):
        rs.sonify(np.arange(8.0), data_duration=1, repeat=0)


@pytest.mark.parametrize(
    ("shape", "method"),
    [
        ((64,), "amplitude"),
        ((64,), "profile"),
        ((32, 16), "erb"),
        ((2, 32, 16), "spatial_erb"),
    ],
)
def test_repeat_works_for_every_dimensionality_and_method(shape, method):
    """repeat 在预处理阶段沿时间轴 tile，因此对所有维度和方法一致可用。"""
    rng = np.random.default_rng(3)
    data = rng.normal(size=shape)

    single = rs.sonify(data, data_duration=0.2, method=method, repeat=1)
    tripled = rs.sonify(data, data_duration=0.2, method=method, repeat=3)

    assert single.output_duration == pytest.approx(0.2)
    assert tripled.output_duration == pytest.approx(0.6)
    assert tripled.repeat == 3


def test_repeat_tiles_the_data_rather_than_the_audio():
    """三遍 repeat 的输出必须是三个逐样本相同的周期。"""
    profile = np.zeros(16)
    profile[4] = 1.0

    result = rs.sonify(
        profile,
        data_duration=0.3,
        method="amplitude",
        repeat=3,
        method_params={"sr": 8_000, "freq": 400.0},
    )

    cycles = result.audio.reshape(3, -1)
    np.testing.assert_allclose(cycles[1], cycles[0], atol=2e-5)
    np.testing.assert_allclose(cycles[2], cycles[0], atol=2e-5)


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
    with pytest.raises(TypeError):
        result.preprocess_params["baseline_operation"] = "divide"
    with pytest.raises(ValueError, match="read-only"):
        result.audio[0] = 123
    with pytest.raises(ValueError):
        result.audio.setflags(write=True)


def test_numeric_array_provenance_uses_an_irreversible_read_only_buffer():
    snapshot = _freeze_value(np.arange(4.0))

    with pytest.raises(ValueError):
        snapshot.setflags(write=True)


def test_unified_api_preprocesses_once_before_method_dispatch(monkeypatch):
    received = []

    def fake_amplitude(data, output, duration, **params):
        del output, params
        received.append(data)
        return np.linspace(-1, 1, round(8_000 * duration)), 8_000

    monkeypatch.setattr(amplitude_module, "amplitude_modulate", fake_amplitude)
    result = rs.sonify(
        np.array([10.0, 12.0, 1_000.0, 11.0]),
        data_duration=0.01,
        repeat=1,
        preprocess_params={
            "baseline_operation": "subtract",
            "baseline_statistic": "median",
            "clip_percentiles": (10, 90),
        },
    )

    assert len(received) == 1
    assert received[0].min() == pytest.approx(0)
    assert received[0].max() == pytest.approx(1)
    assert received[0].flags.writeable is False
    assert result.preprocess_params["baseline_operation"] == "subtract"
    assert result.preprocess_params["baseline_statistic"] == "median"
    assert result.preprocess_params["clip_percentiles"] == (10.0, 90.0)
    # The recorded surface is exactly the documented shared-preprocessing surface,
    # so provenance stays complete when a setting is added or retired.
    assert set(result.preprocess_params) == set(rs.preprocessing_defaults())


def test_hifigan_auto_frames_follow_requested_audio_duration(monkeypatch):
    received = []
    expected_bins = 32
    duration = expected_bins * 256 / 22_050

    def fake_hifigan(data, output, **params):
        received.append((data, output, params))
        samples = expected_bins * 256
        return np.sin(np.linspace(0, 4 * np.pi, samples)), 22_050

    _patch_hifigan_runner(monkeypatch, fake_hifigan)
    time = np.arange(1_000, dtype=np.float64)[:, None]
    feature = np.arange(160, dtype=np.float64)[None, :]
    data = 20 + 0.001 * time + 0.0001 * feature
    data[400:520, 50:110] += 3

    result = rs.sonify(
        data,
        data_duration=duration,
        method="hifigan",
    )

    prepared, output, params = received[0]
    assert prepared.shape == (expected_bins, 160)
    assert prepared.min() == pytest.approx(0)
    assert prepared.max() == pytest.approx(1)
    assert prepared.flags.writeable is False
    assert output is None
    # 除溯源出参外没有任何方法级数据旋钮。
    assert set(params) == {"provenance"}
    assert result.preprocess_params["time_rebin"] == expected_bins
    assert result.preprocess_params["feature_rebin"] is None
    assert result.method_native_samples == expected_bins * 256
    assert result.method_time_scale == pytest.approx(1)


def test_hifigan_auto_frames_do_not_invent_time_bins(monkeypatch):
    received_shapes = []

    def fake_hifigan(data, output, **params):
        del output, params
        received_shapes.append(data.shape)
        return np.sin(np.linspace(0, 4 * np.pi, 2_205)), 22_050

    _patch_hifigan_runner(monkeypatch, fake_hifigan)
    result = rs.sonify(
        np.arange(32.0).reshape(8, 4),
        data_duration=0.1,
        method="hifigan",
    )

    assert received_shapes == [(8, 4)]
    assert result.preprocess_params["time_rebin"] == 8


def test_explicit_preprocess_time_rebin_overrides_method_auto_sizing(monkeypatch):
    received_shapes = []

    def fake_hifigan(data, output, **params):
        del output, params
        received_shapes.append(data.shape)
        return np.sin(np.linspace(0, 4 * np.pi, 2_205)), 22_050

    _patch_hifigan_runner(monkeypatch, fake_hifigan)
    result = rs.sonify(
        np.arange(12.0).reshape(4, 3),
        data_duration=0.1,
        method="hifigan",
        preprocess_params={"time_rebin": 10},
    )

    assert received_shapes == [(10, 3)]
    assert result.preprocess_params["time_rebin"] == 10


def test_invalid_hifigan_auto_frame_value_fails_before_synthesis(monkeypatch):
    monkeypatch.setattr(
        hifigan_module,
        "hifigan",
        lambda *args, **kwargs: pytest.fail("synthesis should not run"),
    )

    with pytest.raises(ValueError, match="time_rebin"):
        rs.sonify(
            np.ones((8, 8)),
            data_duration=1,
            method="hifigan",
            method_params={"time_rebin": "content-aware"},
        )


def test_unified_hifigan_does_not_boost_quiet_generator_output(monkeypatch):
    tone = 0.08 * np.sin(2 * np.pi * 440 * np.arange(22_050) / 22_050)

    def fake_hifigan(data, output, **params):
        del data, output, params
        return tone, 22_050

    _patch_hifigan_runner(monkeypatch, fake_hifigan)
    result = rs.sonify(
        np.arange(64.0).reshape(8, 8),
        data_duration=1,
        method="hifigan",
    )

    assert np.max(np.abs(result.audio)) == pytest.approx(0.08, rel=1e-4)
    assert np.max(np.abs(result.audio)) < 0.1


def test_preprocess_parameter_errors_fail_before_synthesis(monkeypatch):
    monkeypatch.setattr(
        amplitude_module,
        "amplitude_modulate",
        lambda *args, **kwargs: pytest.fail("synthesis should not run"),
    )

    with pytest.raises(ValueError, match="unknown preprocessing"):
        rs.sonify(
            np.ones(8),
            data_duration=1,
            preprocess_params={"clean": True},
        )
    with pytest.raises(ValueError, match="keys must be strings"):
        rs.sonify(
            np.ones(8),
            data_duration=1,
            preprocess_params={1: "median"},
        )


def test_method_validation_finishes_before_preprocessing(monkeypatch):
    monkeypatch.setattr(
        pipeline_module,
        "_preprocess_validated",
        lambda *args, **kwargs: pytest.fail("preprocessing must not run"),
    )

    with pytest.raises(ValueError, match="freq"):
        rs.sonify(
            np.ones(16),
            data_duration=1,
            method="amplitude",
            method_params={"freq": -1},
        )


def test_primary_audio_must_match_the_registered_channel_contract(monkeypatch):
    monkeypatch.setattr(
        amplitude_module,
        "amplitude_modulate",
        lambda *args, **kwargs: (np.zeros((8_000, 2)), 8_000),
    )

    with pytest.raises(RuntimeError, match=r"amplitude.*produced 2 channel.*declares 1"):
        rs.sonify(
            np.ones(16),
            data_duration=1,
            method="amplitude",
            repeat=1,
            method_params={"sr": 8_000},
        )


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
    original = pipeline_module.fit_audio_duration
    calls = []

    def counted_fit(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "fit_audio_duration", counted_fit)
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


def test_output_parent_file_fails_before_synthesis(monkeypatch, tmp_path):
    parent_file = tmp_path / "occupied"
    parent_file.write_bytes(b"file")
    monkeypatch.setattr(
        amplitude_module,
        "amplitude_modulate",
        lambda *args, **kwargs: pytest.fail("synthesis should not run"),
    )

    with pytest.raises(ValueError, match="parent is not a directory"):
        rs.sonify(
            np.ones(16),
            data_duration=1,
            output=parent_file / "result.wav",
        )


def test_result_records_original_input_geometry():
    source = rs.SonificationInput(
        np.arange(24.0).reshape(4, 6),
        duration=0.01,
        time_axis=1,
        name="candidate",
    )

    result = rs.sonify(
        source,
        method="amplitude",
        repeat=1,
        method_params={"sr": 8_000, "freq": 500},
    )

    assert result.input_shape == (4, 6)
    assert result.source_time_axis == 1
    assert result.source_layer_axis is None
    assert result.source_name == "candidate"


def test_griffinlim_feature_geometry_tracks_n_fft(monkeypatch):
    received_shapes = []

    def fake_griffinlim(data, output, **params):
        del output, params
        received_shapes.append(data.shape)
        return np.linspace(-1, 1, 80), 8_000

    monkeypatch.setattr(griffinlim_module, "griffinlim", fake_griffinlim)
    result = rs.sonify(
        np.arange(64.0).reshape(8, 8),
        data_duration=0.01,
        method="griffinlim",
        method_params={"sr": 8_000, "n_fft": 16, "frame_length": 0.001, "n_iter": 1},
    )

    assert received_shapes == [(41, 9)]
    assert result.preprocess_params["feature_rebin"] == 9


def test_unified_griffinlim_supports_small_fft_geometry_end_to_end():
    result = rs.sonify(
        np.arange(64.0).reshape(8, 8),
        data_duration=0.01,
        method="griffinlim",
        repeat=1,
        method_params={"sr": 8_000, "n_fft": 16, "frame_length": 0.001, "n_iter": 1},
    )

    assert result.preprocess_params["feature_rebin"] == 9
    assert result.method_native_samples == 80
    assert result.method_time_scale == pytest.approx(1)
    assert result.audio.shape == (80,)
    assert np.all(np.isfinite(result.audio))


def test_unified_griffinlim_repeat_shares_istft_boundary_frames():
    result = rs.sonify(
        np.arange(64.0).reshape(8, 8),
        data_duration=0.01,
        method="griffinlim",
        repeat=3,
        method_params={"sr": 8_000, "n_fft": 16, "frame_length": 0.001, "n_iter": 1},
    )

    assert result.method_native_samples == 240
    assert result.method_time_scale == pytest.approx(1)


def test_single_griffinlim_frame_is_valid_when_the_timeline_is_not_repeated(monkeypatch):
    received = []

    def fake_griffinlim(data, output, **params):
        del output, params
        received.append(data.shape)
        return np.zeros(80), 8_000

    monkeypatch.setattr(griffinlim_module, "griffinlim", fake_griffinlim)
    result = rs.sonify(
        np.ones((4, 8)),
        data_duration=0.01,
        method="griffinlim",
        repeat=1,
        preprocess_params={"time_rebin": 1, "feature_rebin": 8},
        method_params={
            "sr": 8_000,
            "n_iter": 1,
            "n_fft": 16,
            "frame_length": 0.001,
        },
    )

    assert received == [(1, 8)]
    assert result.output_duration == pytest.approx(0.01)
    assert len(result.audio) == 80


def test_musicnet_preflight_failure_happens_before_primary_synthesis(monkeypatch):
    monkeypatch.setattr(
        amplitude_module,
        "amplitude_modulate",
        lambda *args, **kwargs: pytest.fail("primary synthesis should not run"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_preprocess_validated",
        lambda *args, **kwargs: pytest.fail("preprocessing should not run"),
    )

    def unavailable(**params):
        del params
        raise ImportError("musicnet runtime unavailable")

    monkeypatch.setattr(musicnet_module, "_preflight_musicnet", unavailable)

    with pytest.raises(ImportError, match="runtime unavailable"):
        rs.sonify(np.ones(16), data_duration=1, postprocess="musicnet")


def test_musicnet_receives_planned_primary_length_before_preprocessing(monkeypatch):
    monkeypatch.setattr(
        amplitude_module,
        "amplitude_modulate",
        lambda *args, **kwargs: pytest.fail("primary synthesis should not run"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_preprocess_validated",
        lambda *args, **kwargs: pytest.fail("preprocessing should not run"),
    )
    received = {}

    def reject_short(**params):
        received.update(params)
        raise ValueError("planned MusicNet input is too short")

    monkeypatch.setattr(musicnet_module, "_preflight_musicnet", reject_short)

    with pytest.raises(ValueError, match="planned MusicNet input is too short"):
        rs.sonify(
            np.ones(16),
            data_duration=0.02,
            repeat=1,
            method="amplitude",
            method_params={"sr": 48_000},
            postprocess="musicnet",
        )

    assert received["input_channels"] == 1
    assert received["input_sample_rate"] == 48_000
    assert received["input_samples"] == 960


def test_layer_rebin_is_planned_before_spatial_control_validation(monkeypatch):
    monkeypatch.setattr(
        pipeline_module,
        "_preprocess_validated",
        lambda *args, **kwargs: pytest.fail("preprocessing should not run"),
    )

    with pytest.raises(ValueError, match=r"pan_positions.*two value|pan_positions.*2"):
        rs.sonify(
            np.ones((4, 8, 8)),
            data_duration=1,
            method="spatial_erb",
            preprocess_params={"layer_rebin": 2},
            method_params={"pan_positions": (-1.0, -0.3, 0.3, 1.0)},
        )


def test_spatial_method_receives_the_planned_layer_count(monkeypatch):
    spatial_module = importlib.import_module("radiosonify.spatial")
    received = []

    def fake_spatial(data, output, duration, **params):
        del output, params
        received.append(data.shape)
        return np.zeros((round(8_000 * duration), 2)), 8_000

    monkeypatch.setattr(spatial_module, "spatial_sonify", fake_spatial)
    result = rs.sonify(
        np.arange(256.0).reshape(4, 8, 8),
        data_duration=0.01,
        method="spatial_erb",
        preprocess_params={"layer_rebin": 2},
        method_params={"sr": 8_000, "pan_positions": (-1.0, 1.0)},
    )

    assert received == [(2, 8, 8)]
    assert result.preprocess_params["layer_rebin"] == 2


def test_griffinlim_rejects_feature_bins_above_fft_limit_before_synthesis(
    monkeypatch,
):
    monkeypatch.setattr(
        griffinlim_module,
        "griffinlim",
        lambda *args, **kwargs: pytest.fail("synthesis should not run"),
    )

    with pytest.raises(ValueError, match=r"feature_rebin \(10\).*exceed 9"):
        rs.sonify(
            np.arange(64.0).reshape(8, 8),
            data_duration=0.01,
            method="griffinlim",
            preprocess_params={"feature_rebin": 10},
            method_params={
                "sr": 8_000,
                "n_fft": 16,
                "frame_length": 0.001,
                "n_iter": 1,
            },
        )


def test_fixed_feature_geometry_rejects_conflicting_preprocess_size(monkeypatch):
    monkeypatch.setattr(
        amplitude_module,
        "amplitude_modulate",
        lambda *args, **kwargs: pytest.fail("synthesis should not run"),
    )

    with pytest.raises(ValueError, match="requires feature_rebin=1"):
        rs.sonify(
            np.arange(32.0).reshape(8, 4),
            data_duration=0.01,
            method="amplitude",
            repeat=1,
            preprocess_params={"feature_rebin": 2},
            method_params={"sr": 8_000, "freq": 500},
        )


@pytest.fixture(autouse=True)
def _stub_musicnet_preflight(monkeypatch):
    """API unit tests replace the expensive optional runtime with local fakes."""
    monkeypatch.setattr(musicnet_module, "_preflight_musicnet", lambda **params: None)


def test_provenance_freezes_any_sequence_parameter_by_value():
    """任意 Sequence 参数都必须按值记录，而不是按引用。

    ``pan_positions`` / ``layer_gains`` 的公开契约接受通用 Sequence。曾经的冻结
    逻辑只认 ``list`` / ``tuple``，于是 ``UserList`` 被原样存进结果：调用者随后
    改动原对象，``result.method_params`` 会跟着变。
    """
    pans = UserList([-1.0, 1.0])
    gains = UserList([1.0, 0.5])

    result = rs.sonify(
        np.random.default_rng(0).random((2, 16, 4)),
        data_duration=1.0,
        method="spatial_erb",
        method_params={"pan_positions": pans, "layer_gains": gains},
    )

    assert result.method_params["pan_positions"] == (-1.0, 1.0)
    assert result.method_params["layer_gains"] == (1.0, 0.5)

    pans[0] = 0.25
    gains[1] = 99.0
    assert result.method_params["pan_positions"] == (-1.0, 1.0)
    assert result.method_params["layer_gains"] == (1.0, 0.5)


@pytest.mark.parametrize("name", ["pan_positions", "layer_gains"])
def test_unified_spatial_api_rejects_one_shot_control_iterators(name):
    values = iter([-1.0, 1.0])

    with pytest.raises(ValueError, match=rf"{name} must be a reusable sequence"):
        rs.sonify(
            np.ones((2, 4, 4)),
            data_duration=0.01,
            method="spatial_erb",
            method_params={
                "sr": 8_000,
                "max_freq": 2_000,
                "n_bands": 4,
                name: values,
            },
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("mad", "mad"),
        (b"raw", b"raw"),
        (bytearray(b"raw"), b"raw"),
        (range(3), (0, 1, 2)),
        (UserList([1, UserList([2, 3])]), (1, (2, 3))),
        ({1, 2}, frozenset({1, 2})),
        ((5, 95), (5, 95)),
    ],
)
def test_freeze_provenance_value_records_containers_by_value(value, expected):
    frozen = _freeze_value(value)

    assert frozen == expected
    # Text stays text; recursing into it would turn a choice into characters.
    assert isinstance(frozen, type(expected))


def test_freeze_provenance_value_keeps_nested_mappings_read_only():
    frozen = _freeze_value({"voice_params": UserList([{"detune_cents": 10.0}])})

    nested = frozen["voice_params"][0]
    with pytest.raises(TypeError):
        nested["detune_cents"] = 0.0


def test_grouped_method_parameters_are_recorded_at_full_resolution():
    """分组参数与顶层参数使用同一种溯源口径。

    顶层 method_params 记录全部注册默认值，分组以前只记调用者传入的子集
    （省略时甚至只是 None），于是同一条记录里出现两种口径。
    """
    from radiosonify._perceptual_config import EVENT_DEFAULTS, VOICE_DEFAULTS

    matrix = np.random.default_rng(0).random((16, 8))
    supplied = rs.sonify(
        matrix,
        data_duration=0.1,
        method="erb",
        method_params={"voice_params": {"harmonic_limit_hz": 6000.0}},
    )
    omitted = rs.sonify(matrix, data_duration=0.1, method="erb")

    assert set(supplied.method_params["voice_params"]) == set(VOICE_DEFAULTS)
    assert set(supplied.method_params["event_params"]) == set(EVENT_DEFAULTS)
    assert supplied.method_params["voice_params"]["harmonic_limit_hz"] == 6000.0
    assert supplied.method_params["voice_params"]["detune_cents"] == VOICE_DEFAULTS["detune_cents"]
    assert dict(omitted.method_params["voice_params"]) == dict(VOICE_DEFAULTS)
    assert dict(omitted.method_params["event_params"]) == dict(EVENT_DEFAULTS)


def test_resolved_plan_recursively_freezes_grouped_parameters():
    planning_module = importlib.import_module("radiosonify.planning")
    plan = planning_module.resolve_sonification_plan(
        np.ones((8, 8)),
        data_duration=0.1,
        data_type=None,
        method="erb",
        speed=1.0,
        repeat=1,
        preserve_pitch=False,
        output_sr=None,
        preprocess_params=None,
        method_params={"voice_params": {"fm_index": 0.5}},
        postprocess=None,
        postprocess_params=None,
        output=None,
    )

    with pytest.raises(TypeError):
        plan.method_params["voice_params"]["fm_index"] = 1.0


def test_misspelled_grouped_key_fails_before_synthesis_starts(monkeypatch):
    calls = []
    original = erb_module.erb_sonify

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(erb_module, "erb_sonify", spy)
    with pytest.raises(ValueError, match="unknown voice_params key"):
        rs.sonify(
            np.random.default_rng(0).random((16, 8)),
            data_duration=0.1,
            method="erb",
            method_params={"voice_params": {"detune_cent": 5}},
        )

    assert calls == []
