import json
import sys
import types
import numpy as np
import pytest

import radiosonify.hifigan as hifigan_module


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


class _FakeTorch:
    cuda = _FakeCuda()

    @staticmethod
    def manual_seed(seed):
        return None

    @staticmethod
    def device(name):
        return name

    @staticmethod
    def load(path, map_location=None):
        return {"generator": {}}

    @staticmethod
    def no_grad():
        return _FakeNoGrad()

    @staticmethod
    def FloatTensor(x):
        return _FakeTensor(x)


class _FakeAttrDict(dict[str, object]):
    def __getattr__(self, item):
        return self[item]


class _FakeGenerator:
    def __init__(self, cfg):
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
    setattr(fake_env_module, "AttrDict", _FakeAttrDict)
    fake_generator_module = types.ModuleType("radiosonify.models.hifigan.generator")
    setattr(fake_generator_module, "Generator", _FakeGenerator)

    monkeypatch.setitem(sys.modules, "radiosonify.models.hifigan.env", fake_env_module)
    monkeypatch.setitem(sys.modules, "radiosonify.models.hifigan.generator", fake_generator_module)


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
        monkeypatch.setattr(hifigan_module, "_require_torch", lambda: _FakeTorch)
        monkeypatch.setattr(hifigan_module, "_require_skimage", lambda: (lambda x, y: x))

        with pytest.raises(ValueError, match="2D"):
            hifigan_module.hifigan(np.ones(128))

    @pytest.mark.filterwarnings("ignore:weights_only=True not supported")
    def test_returns_audio_and_sr(self, fake_hifigan_runtime):
        spec = np.random.default_rng(42).random((256, 1024))
        audio, sr = hifigan_module.hifigan(spec, time_rebin=128)
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 1e-6
        assert sr == 48000

    @pytest.mark.filterwarnings("ignore:weights_only=True not supported")
    def test_saves_to_file(self, fake_hifigan_runtime, tmp_path):
        spec = np.random.default_rng(42).random((256, 1024))
        out = tmp_path / "hifigan.wav"
        audio, sr = hifigan_module.hifigan(spec, time_rebin=128, output=str(out))
        assert out.exists()
        assert out.stat().st_size > 0
