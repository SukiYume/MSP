import numpy as np
import pytest
import soundfile as sf

import radiosonify as rs
import radiosonify.spatial as spatial_module
from radiosonify._perceptual import _true_peak
from radiosonify.spatial import spatial_sonify


def test_spatial_sonify_returns_stereo_with_strict_duration(tmp_path):
    cube = np.zeros((2, 4, 4))
    cube[0, :, 0] = 1
    cube[1, :, -1] = 1
    output = tmp_path / "layers.wav"

    audio, sr = spatial_sonify(
        cube,
        sr=8_000,
        duration=0.1,
        min_freq=300,
        max_freq=2_000,
        n_bands=4,
        pan_positions=(-1, 1),
        output=output,
    )

    assert audio.shape == (800, 2)
    assert audio.dtype == np.float32
    assert _true_peak(audio) <= 10 ** (-1 / 20) + 1e-5
    assert np.sqrt(np.mean(audio**2)) <= 10 ** (-20 / 20) + 1e-6
    wav, wav_sr = sf.read(output, always_2d=True)
    assert wav.shape == (800, 2)
    assert wav_sr == sr


def test_extreme_pan_places_a_single_layer_on_one_side():
    cube = np.zeros((2, 4, 4))
    cube[0, :, 0] = 1

    audio, _ = spatial_sonify(
        cube,
        sr=8_000,
        duration=0.1,
        max_freq=2_000,
        n_bands=4,
        pan_positions=(-1, 1),
    )

    assert np.sqrt(np.mean(audio[:, 0] ** 2)) > 0.005
    assert np.max(np.abs(audio[:, 1])) < 1e-6


def test_declaring_layer_axis_on_the_input_gives_identical_audio():
    """channels-last 的立方体声明 layer_axis 后必须与 layer-first 逐样本一致。"""
    rng = np.random.default_rng(11)
    layers_first = rng.normal(size=(3, 24, 5))
    channels_last = np.moveaxis(layers_first, 0, -1)

    first = rs.sonify(
        layers_first,
        data_duration=0.05,
        method_params={"max_freq": 2_000, "n_bands": 5},
    )
    last = rs.sonify(
        rs.SonificationInput(channels_last, duration=0.05, layer_axis=-1),
        method_params={"max_freq": 2_000, "n_bands": 5},
    )

    np.testing.assert_allclose(first.audio, last.audio)


def test_preprocessed_layers_share_the_same_continuous_amplitude_mapping():
    cube = np.zeros((2, 4, 4))
    cube[0, :, 0] = 1
    cube[1, :, 0] = 0.25

    audio, _ = spatial_sonify(
        cube,
        sr=8_000,
        duration=0.1,
        max_freq=2_000,
        n_bands=4,
        pan_positions=(-1, 1),
    )
    left_rms = np.sqrt(np.mean(audio[:, 0] ** 2))
    right_rms = np.sqrt(np.mean(audio[:, 1] ** 2))

    assert left_rms / right_rms == pytest.approx(4**4, rel=0.02)


def test_spatial_retro_timbre_is_deterministic_and_differs_from_sine():
    cube = np.full((2, 64, 8), 0.2)
    cube[0, 28:34, 3:5] = 1.0
    settings = {
        "sr": 8_000,
        "duration": 0.25,
        "max_freq": 2_000,
        "n_bands": 8,
        "pan_positions": (-1, 1),
    }

    retro, _ = spatial_sonify(cube, timbre="retro_digital", **settings)
    repeated, _ = spatial_sonify(cube, timbre="retro_digital", **settings)
    sine, _ = spatial_sonify(cube, timbre="sine", **settings)

    np.testing.assert_array_equal(retro, repeated)
    assert not np.allclose(retro, sine)


def test_spatial_layers_share_one_global_event_density_budget(monkeypatch):
    rate_scales = []

    def synthesize(data, *, settings, event_rate_scale=1.0):
        rate_scales.append(event_rate_scale)
        return np.zeros(round(settings.sr * settings.duration))

    monkeypatch.setattr(spatial_module, "_synthesize_prepared", synthesize)

    spatial_sonify(
        np.ones((3, 4, 4)),
        sr=8_000,
        duration=0.1,
        max_freq=2_000,
        event_voice="water_drop",
    )

    assert rate_scales == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert sum(rate_scales) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pan_positions": [0]}, "one value per layer"),
        ({"pan_positions": [-2, 0]}, "pan_positions"),
        ({"layer_gains": [1, -1]}, "layer_gains"),
    ],
)
def test_spatial_parameter_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        spatial_sonify(np.ones((2, 4, 4)), duration=0.01, **kwargs)


@pytest.mark.parametrize("name", ["pan_positions", "layer_gains"])
def test_spatial_rejects_one_shot_control_iterators_before_synthesis(monkeypatch, name):
    monkeypatch.setattr(
        spatial_module,
        "_synthesize_prepared",
        lambda *args, **kwargs: pytest.fail("synthesis should not run"),
    )

    with pytest.raises(ValueError, match=rf"{name} must be a reusable sequence"):
        spatial_sonify(
            np.ones((2, 4, 4)),
            duration=0.01,
            **{name: iter([-1.0, 1.0])},
        )


def test_spatial_accepts_numpy_control_arrays():
    audio, _ = spatial_sonify(
        np.ones((2, 4, 4)),
        sr=8_000,
        duration=0.01,
        max_freq=2_000,
        n_bands=4,
        pan_positions=np.array([-1.0, 1.0]),
        layer_gains=np.array([1.0, 0.5]),
    )

    assert audio.shape == (80, 2)


def test_layered_matrix_auto_method_and_duration_contract(tmp_path):
    output = tmp_path / "iquv.wav"
    result = rs.sonify(
        np.arange(4 * 4 * 4, dtype=float).reshape(4, 4, 4),
        data_duration=0.02,
        data_type="iquv",
        method_params={"sr": 8_000, "max_freq": 3_000, "n_bands": 4},
        output=output,
    )

    assert result.data_type is rs.DataType.LAYERED_MATRIX
    assert result.method == "spatial_erb"
    assert result.audio.shape == (160, 2)
    assert result.output_duration == pytest.approx(0.02)
    assert sf.info(output).channels == 2


def test_spatial_parameter_provenance_is_an_immutable_snapshot():
    pans = [-1.0, 1.0]
    gains = [1.0, 0.5]
    result = rs.sonify(
        np.ones((2, 2, 2)),
        data_duration=0.01,
        method_params={
            "sr": 8_000,
            "max_freq": 3_000,
            "n_bands": 2,
            "pan_positions": pans,
            "layer_gains": gains,
        },
    )

    pans[0] = 0.0
    gains.append(0.25)
    assert result.method_params["pan_positions"] == (-1.0, 1.0)
    assert result.method_params["layer_gains"] == (1.0, 0.5)
    with pytest.raises(TypeError):
        result.method_params["pan_positions"] = (0.0, 0.0)
