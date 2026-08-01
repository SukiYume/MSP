import importlib
import json
import sys
import types

import numpy as np
import pytest

import radiosonify as rs

hifigan_module = importlib.import_module("radiosonify.hifigan")
core_module = importlib.import_module("radiosonify.core")


def test_rejects_bad_output_path_before_spectrogram_preparation(monkeypatch):
    monkeypatch.setattr(
        hifigan_module,
        "_prepare_spectrogram",
        lambda *args, **kwargs: pytest.fail("spectrogram preparation must not run"),
    )

    with pytest.raises(ValueError, match=".wav"):
        hifigan_module.hifigan(np.ones((8, 8)), output="bad.flac")


def test_preprocessing_scans_input_finiteness_once(monkeypatch):
    calls = []
    original_isfinite = core_module.np.isfinite

    def tracked_isfinite(value):
        calls.append(np.asarray(value).shape)
        return original_isfinite(value)

    monkeypatch.setattr(core_module.np, "isfinite", tracked_isfinite)
    prepared = hifigan_module._prepare_spectrogram(
        np.arange(256, dtype=np.float64).reshape(16, 16),
        time_rebin=8,
        time_smoothing=None,
        clean=True,
        exposure_cut=25,
    )

    assert prepared.shape == (8, 16)
    assert calls == [(16, 16)]


class _FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTensor:
    def __init__(self, data):
        self._data = np.array(data, dtype=np.float32)

    def to(self, device):
        return self

    def squeeze(self):
        return self

    def reshape(self, *shape):
        self._data = self._data.reshape(*shape)
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class _FakeCuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def manual_seed(seed):
        return None

    @staticmethod
    def current_device():
        return 0


class _FakeDevice:
    def __init__(self, name):
        self.type = name
        self.index = None


class _FakeRandom:
    @staticmethod
    def fork_rng(devices):
        return _FakeNoGrad()


class _FakeTorch:
    cuda = _FakeCuda()
    random = _FakeRandom()
    float32 = np.float32

    @staticmethod
    def manual_seed(seed):
        return None

    @staticmethod
    def device(name):
        return _FakeDevice(name)

    @staticmethod
    def load(path, map_location=None, weights_only=False):
        return {"generator": {}}

    @staticmethod
    def inference_mode():
        return _FakeNoGrad()

    @staticmethod
    def as_tensor(x, dtype=None, device=None):
        return _FakeTensor(x)


class _FakeAttrDict(dict[str, object]):
    def __getattr__(self, item):
        return self[item]


class _FakeGenerator:
    init_count = 0

    def __init__(self, cfg):
        type(self).init_count += 1
        self.cfg = cfg

    def to(self, device):
        return self

    def load_state_dict(self, state):
        return None

    def eval(self):
        return self

    def remove_weight_norm(self):
        return None

    def __call__(self, x_tensor):
        return _FakeTensor(np.sin(np.linspace(0, 4 * np.pi, 1024, dtype=np.float32)))


@pytest.fixture
def fake_hifigan_runtime(monkeypatch, tmp_path):
    hifigan_module._load_generator.cache_clear()
    _FakeGenerator.init_count = 0
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "generator.pth"
    config_path.write_text(json.dumps({"seed": 123, "sampling_rate": 48000}), encoding="utf-8")
    checkpoint_path.write_text("fake", encoding="utf-8")

    monkeypatch.setattr(hifigan_module, "_require_torch", lambda: _FakeTorch)

    def _fake_resize(data, shape):
        target_t, target_f = shape
        return np.resize(data, (target_t, target_f))

    monkeypatch.setattr(hifigan_module, "_require_skimage", lambda: _fake_resize)

    def _fake_get_model_path(model, filename):
        if filename == "config.json":
            return str(config_path)
        if filename == "generator.pth":
            return str(checkpoint_path)
        raise AssertionError(f"unexpected filename: {filename}")

    monkeypatch.setattr(hifigan_module, "get_model_path", _fake_get_model_path)

    fake_env_module = types.ModuleType("radiosonify.models.hifigan.env")
    fake_env_module.AttrDict = _FakeAttrDict
    fake_generator_module = types.ModuleType("radiosonify.models.hifigan.generator")
    fake_generator_module.Generator = _FakeGenerator

    monkeypatch.setitem(sys.modules, "radiosonify.models.hifigan.env", fake_env_module)
    monkeypatch.setitem(sys.modules, "radiosonify.models.hifigan.generator", fake_generator_module)
    yield
    hifigan_module._load_generator.cache_clear()


class TestHifiGAN:
    def test_rescale_data_shape(self):
        spec = np.random.default_rng(42).random((128, 512))

        def fake_resize(data, shape):
            return np.resize(data, shape)

        out = hifigan_module._rescale_data(spec, fake_resize)
        assert out.shape == (1, 80, 128)

    @pytest.mark.parametrize(
        "spec",
        [
            np.zeros((64, 128), dtype=np.float64),
            np.pad(np.array([[1.0]], dtype=np.float64), ((0, 63), (0, 127))),
            np.random.default_rng(123).lognormal(mean=0.0, sigma=4.0, size=(64, 128)),
        ],
    )
    def test_rescale_data_handles_extreme_distributions(self, spec):
        def fake_resize(data, shape):
            return np.resize(data, shape)

        out = hifigan_module._rescale_data(spec, fake_resize)
        assert out.shape == (1, 80, spec.shape[0])
        assert np.all(np.isfinite(out))
        assert out.min() >= -11.0
        assert out.max() <= 1.6

    def test_rejects_1d_input(self, monkeypatch):
        with pytest.raises(ValueError, match="2D"):
            hifigan_module.hifigan(np.ones(128))

    def test_rejects_non_finite_input_before_loading_optional_dependencies(self):
        spec = np.ones((4, 4))
        spec[0, 0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            hifigan_module.hifigan(spec)

    def test_rejects_invalid_rebin_before_loading_optional_dependencies(self):
        with pytest.raises(ValueError, match="time_bins"):
            hifigan_module.hifigan(np.ones((4, 4)), time_rebin=5)

    def test_rejects_non_boolean_clean_before_loading_optional_dependencies(self):
        with pytest.raises(ValueError, match="clean"):
            hifigan_module.hifigan(np.ones((4, 4)), clean="no")

    def test_rejects_exposure_cut_even_when_clean_is_disabled(self):
        with pytest.raises(ValueError, match="exposure_cut"):
            hifigan_module.hifigan(np.ones((4, 4)), clean=False, exposure_cut=1)

    @pytest.mark.parametrize("time_smoothing", [0, -1, np.inf, True, "0.75"])
    def test_rejects_invalid_time_smoothing_before_loading_model(self, time_smoothing):
        with pytest.raises(ValueError, match="time_smoothing"):
            hifigan_module.hifigan(
                np.ones((4, 4)),
                time_smoothing=time_smoothing,
            )

    def test_time_smoothing_only_uses_time_axis(
        self,
        fake_hifigan_runtime,
        monkeypatch,
    ):
        calls = []

        def fake_filter(data, sigma, axis, mode):
            calls.append((data.shape, sigma, axis, mode))
            return data

        monkeypatch.setattr(hifigan_module, "gaussian_filter1d", fake_filter)
        hifigan_module.hifigan(
            np.random.default_rng(42).random((32, 64)),
            time_smoothing=0.75,
        )

        assert calls == [((32, 64), 0.75, 0, "reflect")]

    def test_returns_audio_and_sr(self, fake_hifigan_runtime):
        spec = np.random.default_rng(42).random((256, 1024))
        audio, sr = hifigan_module.hifigan(spec, time_rebin=128)
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 1e-6
        assert np.max(np.abs(audio)) <= 0.9 + 1e-6
        assert sr == 48000

    def test_reuses_loaded_generator(self, fake_hifigan_runtime):
        spec = np.random.default_rng(42).random((256, 1024))

        hifigan_module.hifigan(spec, time_rebin=128)
        hifigan_module.hifigan(spec, time_rebin=128)

        assert _FakeGenerator.init_count == 1

    def test_top_level_neural_method_remains_callable_after_first_call(
        self,
        fake_hifigan_runtime,
    ):
        spec = np.random.default_rng(42).random((32, 64))

        rs.hifigan(spec)
        assert callable(rs.hifigan)
        rs.hifigan(spec)
        assert callable(rs.hifigan)

    def test_saves_to_file(self, fake_hifigan_runtime, tmp_path):
        spec = np.random.default_rng(42).random((256, 1024))
        out = tmp_path / "hifigan.wav"
        audio, sr = hifigan_module.hifigan(spec, time_rebin=128, output=str(out))
        assert out.exists()
        assert out.stat().st_size > 0
