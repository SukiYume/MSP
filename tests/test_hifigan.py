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
        np.arange(256, dtype=np.float64).reshape(16, 16) / 255,
        time_rebin=8,
        time_smoothing=None,
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


def test_checkpoint_type_error_never_falls_back_to_unsafe_pickle_loading():
    class FailingTorch:
        calls = []

        @classmethod
        def load(cls, path, **kwargs):
            cls.calls.append((path, kwargs))
            raise TypeError("checkpoint decoding failed")

    with pytest.raises(TypeError, match="decoding failed"):
        hifigan_module._torch_load_state_dict(FailingTorch, "checkpoint.pth", "cpu")

    assert FailingTorch.calls == [("checkpoint.pth", {"map_location": "cpu", "weights_only": True})]


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

        out, offset = hifigan_module._rescale_data(spec, fake_resize)
        assert out.shape == (1, 80, 128)
        assert isinstance(offset, float)

    def test_model_adapter_owns_80_bin_resize_and_range_restoration(self):
        spec = np.tile(np.linspace(0.2, 0.8, 512), (16, 1))
        calls = []

        def fake_resize(data, shape):
            calls.append((data.shape, shape))
            return np.resize(data, shape)

        original, _ = hifigan_module._rescale_data(spec, fake_resize)
        half_scale, _ = hifigan_module._rescale_data(spec * 0.5, fake_resize)

        assert calls == [((16, 512), (16, 80)), ((16, 512), (16, 80))]
        assert np.allclose(original, half_scale)

    def test_model_adapter_matches_historical_checkpoint_mapping(self):
        """Lock the published resize/histogram/magic-number transform verbatim."""
        spec = np.random.default_rng(7).random((32, 512))

        def fake_resize(data, shape):
            return np.resize(data, shape)

        resized = fake_resize(spec, (spec.shape[0], 80))
        resized = (resized - resized.min()) / (resized.max() - resized.min())
        h, w = resized.shape
        histogram = np.histogram(resized.ravel(), bins=int(h * w / 100))
        centres = (histogram[1][1:] + histogram[1][:-1]) / 2
        expected_offset = 0.6 - centres[np.argmax(histogram[0])]
        expected = np.clip((resized + expected_offset) * 12 - 10.5, -11, 1.6)
        expected = expected.T[np.newaxis, :, :]

        actual, actual_offset = hifigan_module._rescale_data(spec, fake_resize)

        assert actual_offset == pytest.approx(expected_offset)
        assert np.array_equal(actual, expected)

    def test_histogram_offset_is_reported_for_provenance(self):
        """直方图偏移随输入分布变化，是这一步唯一的数据依赖量，必须能被记录。"""

        def fake_resize(data, shape):
            return np.resize(data, shape)

        low = np.zeros((16, 80))
        low[:, -1] = 1
        high = np.ones((16, 80))
        high[:, 0] = 0

        _, low_offset = hifigan_module._rescale_data(low, fake_resize)
        _, high_offset = hifigan_module._rescale_data(high, fake_resize)

        assert low_offset != high_offset

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

        out, _ = hifigan_module._rescale_data(spec, fake_resize)
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
        with pytest.raises(ValueError, match="time_rebin"):
            hifigan_module.hifigan(np.ones((4, 4)), time_rebin=0)

    def test_direct_low_level_rebin_can_upsample(self):
        prepared = hifigan_module._prepare_spectrogram(
            np.arange(16.0).reshape(4, 4) / 15,
            time_rebin=9,
            time_smoothing=None,
        )

        assert prepared.shape == (9, 4)
        assert prepared.min() >= 0
        assert prepared.max() <= 1

    def test_rejects_data_that_skipped_shared_preprocessing(self):
        with pytest.raises(ValueError, match="preprocess"):
            hifigan_module.hifigan(np.arange(16.0).reshape(4, 4))

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
        assert np.max(np.abs(audio)) <= 1.0 + 1e-6
        assert np.max(np.abs(audio)) > 0.99
        assert sr == 48000

    def test_does_not_peak_normalize_quiet_generator_output(
        self,
        fake_hifigan_runtime,
        monkeypatch,
    ):
        def quiet_generator(self, x_tensor):
            del self, x_tensor
            values = 0.08 * np.sin(np.linspace(0, 4 * np.pi, 1024, dtype=np.float32))
            return _FakeTensor(values)

        monkeypatch.setattr(_FakeGenerator, "__call__", quiet_generator)
        audio, _ = hifigan_module.hifigan(np.ones((32, 80)))

        assert np.max(np.abs(audio)) == pytest.approx(0.08, rel=1e-4)

    def test_reuses_loaded_generator(self, fake_hifigan_runtime):
        spec = np.random.default_rng(42).random((256, 1024))

        hifigan_module.hifigan(spec, time_rebin=128)
        hifigan_module.hifigan(spec, time_rebin=128)

        assert _FakeGenerator.init_count == 1

    def test_top_level_neural_alias_remains_callable_after_first_call(
        self,
        fake_hifigan_runtime,
    ):
        spec = np.random.default_rng(42).random((32, 64))

        rs.hifigan_vocode(spec)
        assert callable(rs.hifigan_vocode)
        rs.hifigan_vocode(spec)
        assert callable(rs.hifigan_vocode)

    def test_saves_to_file(self, fake_hifigan_runtime, tmp_path):
        spec = np.random.default_rng(42).random((256, 1024))
        out = tmp_path / "hifigan.wav"
        audio, sr = hifigan_module.hifigan(spec, time_rebin=128, output=str(out))
        assert out.exists()
        assert out.stat().st_size > 0
