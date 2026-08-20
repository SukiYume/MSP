from __future__ import annotations

import importlib

import numpy as np
import pytest

import radiosonify as rs

rave_module = importlib.import_module("radiosonify.rave")


class _FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    def to(self, _device):
        return self

    def unsqueeze(self, axis):
        return _FakeTensor(np.expand_dims(self.data, axis))

    def repeat(self, *repeats):
        return _FakeTensor(np.tile(self.data, repeats))

    def detach(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.data


class _FakeModel:
    def __init__(self, sr=8_000, n_channels=1):
        self.sr = sr
        self.n_channels = n_channels
        self.last_input_shape = None
        self.last_input_peak = None

    def eval(self):
        return self

    def forward(self, tensor):
        self.last_input_shape = tensor.data.shape
        self.last_input_peak = float(np.max(np.abs(tensor.data)))
        return _FakeTensor(tensor.data * 0.5)


class _WrongChannelModel(_FakeModel):
    def forward(self, tensor):
        self.last_input_shape = tensor.data.shape
        return _FakeTensor(
            np.zeros((1, self.n_channels + 1, tensor.data.shape[-1]), dtype=np.float32)
        )


class _FakeJit:
    def __init__(self, model):
        self.model = model

    def load(self, _path, map_location):
        assert map_location == "cpu"
        return self.model


class _FakeAvailability:
    @staticmethod
    def is_available():
        return False


class _FakeCudaAvailable:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1


class _InferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeTorch:
    cuda = _FakeAvailability()

    class backends:
        mps = _FakeAvailability()

    def __init__(self, model):
        self.jit = _FakeJit(model)

    @staticmethod
    def device(name):
        return name

    @staticmethod
    def from_numpy(data):
        return _FakeTensor(data)

    @staticmethod
    def inference_mode():
        return _InferenceMode()


def _install_fake_torch(monkeypatch, model):
    fake_torch = _FakeTorch(model)
    monkeypatch.setattr(rave_module, "require", lambda module, extra: fake_torch)
    return fake_torch


def test_rave_mono_model_processes_stereo_channels_independently(monkeypatch, tmp_path):
    model_path = tmp_path / "trusted.ts"
    model_path.write_bytes(b"test placeholder")
    model = _FakeModel(n_channels=1)
    _install_fake_torch(monkeypatch, model)
    stereo = np.column_stack((np.linspace(-0.5, 0.5, 32), np.linspace(0.2, -0.2, 32)))

    transformed, sr = rave_module.rave(
        stereo,
        sr=8_000,
        model_path=model_path,
    )

    assert sr == 8_000
    assert transformed.shape == stereo.shape
    np.testing.assert_allclose(transformed, stereo * 0.5, atol=1e-7)
    assert model.last_input_shape == (2, 1, 32)


def test_rave_resamples_to_model_rate(monkeypatch, tmp_path):
    model_path = tmp_path / "trusted.ts"
    model_path.write_bytes(b"test placeholder")
    model = _FakeModel(sr=8_000)
    _install_fake_torch(monkeypatch, model)

    transformed, sr = rave_module.rave(
        np.sin(np.linspace(0, 2 * np.pi, 40)),
        sr=4_000,
        model_path=model_path,
    )

    assert sr == 8_000
    assert len(transformed) == 80
    assert model.last_input_shape == (1, 1, 80)


def test_rave_limits_resampling_overshoot_before_model_inference(monkeypatch, tmp_path):
    model_path = tmp_path / "trusted.ts"
    model_path.write_bytes(b"test placeholder")
    model = _FakeModel(sr=8_000)
    _install_fake_torch(monkeypatch, model)
    alternating = np.resize(np.array([-1.0, 1.0]), 64)

    rave_module.rave(alternating, sr=4_000, model_path=model_path)

    assert model.last_input_peak <= 1.0


def test_rave_rejects_model_output_with_wrong_channel_count(monkeypatch, tmp_path):
    model_path = tmp_path / "trusted.ts"
    model_path.write_bytes(b"test placeholder")
    model = _WrongChannelModel(n_channels=2)
    _install_fake_torch(monkeypatch, model)

    with pytest.raises(ValueError, match="batch/channel"):
        rave_module.rave(
            np.zeros((32, 2)),
            sr=8_000,
            model_path=model_path,
        )


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_rave_model_integer_metadata_is_strict(value):
    model = _FakeModel()
    model.sr = value

    with pytest.raises(ValueError, match="positive 'sr'"):
        rave_module._model_int(model, "sr")


def test_rave_output_parent_is_checked_before_model_loading(monkeypatch, tmp_path):
    model_path = tmp_path / "trusted.ts"
    model_path.write_bytes(b"test placeholder")
    parent_file = tmp_path / "occupied"
    parent_file.write_bytes(b"file")
    monkeypatch.setattr(
        rave_module,
        "require",
        lambda *args, **kwargs: pytest.fail("Torch must not load"),
    )

    with pytest.raises(ValueError, match="parent is not a directory"):
        rave_module.rave(
            np.zeros(32),
            sr=8_000,
            model_path=model_path,
            output=parent_file / "result.wav",
        )


@pytest.mark.parametrize(
    ("model_name", "device", "message"),
    [
        (None, "auto", "requires model_path"),
        ("missing.ts", "auto", "does not exist"),
        ("model.bin", "auto", "TorchScript"),
        ("model.ts", "quantum", "device"),
        ("model.ts", "cuda:", "device"),
        ("model.ts", "cuda:nope", "device"),
        ("model.ts", "cuda:-1", "device"),
    ],
)
def test_rave_parameter_validation(tmp_path, model_name, device, message):
    model_path = None if model_name is None else tmp_path / model_name
    if model_name in {"model.bin", "model.ts"}:
        model_path.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match=message):
        rave_module._validate_rave_parameters(model_path, device)


def test_unified_api_records_rave_model_provenance(monkeypatch, tmp_path):
    model_path = tmp_path / "style.ts"
    model_path.write_bytes(b"placeholder")
    calls = []

    def fake_rave(input_audio, sr, output, **params):
        calls.append((input_audio.shape, sr, output, params))
        return np.asarray(input_audio), sr

    _install_fake_torch(monkeypatch, _FakeModel())
    monkeypatch.setattr(rave_module, "rave", fake_rave)
    result = rs.sonify(
        np.linspace(0, 1, 16),
        data_duration=0.02,
        repeat=1,
        method_params={"sr": 8_000, "freq": 500},
        postprocess="rave",
        postprocess_params={"model_path": model_path},
    )

    assert result.postprocess == "rave"
    assert result.postprocess_params["model_path"] == str(model_path)
    assert calls[0][2] is None
    assert calls[0][3]["device"] == "auto"


def test_unavailable_rave_dependency_fails_before_primary_synthesis(monkeypatch, tmp_path):
    model_path = tmp_path / "style.ts"
    model_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        importlib.import_module("radiosonify.amplitude"),
        "amplitude_modulate",
        lambda *args, **kwargs: pytest.fail("primary synthesis must not run"),
    )

    def unavailable(*args, **kwargs):
        raise ImportError("torch unavailable")

    monkeypatch.setattr(rave_module, "require", unavailable)
    with pytest.raises(ImportError, match="torch unavailable"):
        rs.sonify(
            np.linspace(0, 1, 16),
            data_duration=0.02,
            postprocess="rave",
            postprocess_params={"model_path": model_path},
        )


def test_nonexistent_cuda_index_fails_during_parameter_validation(monkeypatch, tmp_path):
    model_path = tmp_path / "style.ts"
    model_path.write_bytes(b"placeholder")
    fake_torch = _install_fake_torch(monkeypatch, _FakeModel())
    fake_torch.cuda = _FakeCudaAvailable()

    with pytest.raises(ValueError, match="detected 1 CUDA device"):
        rave_module._validate_rave_parameters(model_path, "cuda:1")
