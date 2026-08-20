from __future__ import annotations

import importlib

import numpy as np
import pytest

import radiosonify as rs

rave_module = importlib.import_module("radiosonify.rave")
pipeline_module = importlib.import_module("radiosonify.pipeline")

try:
    import torch as _real_torch
except ImportError:  # pragma: no cover - exercised by the base-only CI jobs
    _real_torch = None


if _real_torch is not None:

    class _TinyRaveExport(_real_torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sampling_rate = 8_000
            self.register_buffer(
                "forward_params",
                _real_torch.tensor([1, 1, 1, 1], dtype=_real_torch.int64),
            )

        def forward(self, audio):
            return audio * 0.25


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
    def __init__(
        self,
        sampling_rate=8_000,
        input_channels=1,
        output_channels=None,
        input_divider=1,
        output_divider=1,
    ):
        output_channels = input_channels if output_channels is None else output_channels
        self.sampling_rate = sampling_rate
        self.forward_params = np.asarray(
            [input_channels, input_divider, output_channels, output_divider],
            dtype=np.int64,
        )
        self.output_channels = output_channels
        self.last_input_shape = None
        self.last_input_peak = None

    def eval(self):
        return self

    def forward(self, tensor):
        self.last_input_shape = tensor.data.shape
        self.last_input_peak = float(np.max(np.abs(tensor.data)))
        transformed = tensor.data * 0.5
        if transformed.shape[1] == 1 and self.output_channels > 1:
            transformed = np.repeat(transformed, self.output_channels, axis=1)
        return _FakeTensor(transformed)


class _WrongChannelModel(_FakeModel):
    def forward(self, tensor):
        self.last_input_shape = tensor.data.shape
        return _FakeTensor(
            np.zeros((1, self.output_channels + 1, tensor.data.shape[-1]), dtype=np.float32)
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


class _FakeDefaultGenerator:
    seeds = []

    @classmethod
    def manual_seed(cls, seed):
        cls.seeds.append(seed)


class _FakeRandom:
    default_generator = _FakeDefaultGenerator()

    @staticmethod
    def fork_rng(devices):
        assert devices == []
        return _InferenceMode()


class _FakeTorch:
    cuda = _FakeAvailability()
    random = _FakeRandom()

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
    _FakeDefaultGenerator.seeds.clear()
    fake_torch = _FakeTorch(model)
    monkeypatch.setattr(rave_module, "require", lambda module, extra: fake_torch)
    return fake_torch


@pytest.fixture
def exported_model_path(tmp_path):
    path = tmp_path / "trusted.ts"
    path.write_bytes(b"test placeholder")
    return path


def test_rave_mono_model_processes_stereo_channels_independently(monkeypatch, exported_model_path):
    model = _FakeModel(input_channels=1)
    _install_fake_torch(monkeypatch, model)
    stereo = np.column_stack((np.linspace(-0.5, 0.5, 32), np.linspace(0.2, -0.2, 32)))

    transformed, sr = rave_module.rave(
        stereo,
        sr=8_000,
        model_path=exported_model_path,
    )

    assert sr == 8_000
    assert transformed.shape == stereo.shape
    np.testing.assert_allclose(transformed, stereo * 0.5, atol=1e-7)
    assert model.last_input_shape == (2, 1, 32)


def test_rave_resamples_to_model_rate(monkeypatch, exported_model_path):
    model = _FakeModel(sampling_rate=8_000)
    _install_fake_torch(monkeypatch, model)

    transformed, sr = rave_module.rave(
        np.sin(np.linspace(0, 2 * np.pi, 40)),
        sr=4_000,
        model_path=exported_model_path,
    )

    assert sr == 8_000
    assert len(transformed) == 80
    assert model.last_input_shape == (1, 1, 80)


def test_rave_supports_standard_mono_input_stereo_output(monkeypatch, exported_model_path):
    model = _FakeModel(input_channels=1, output_channels=2)
    _install_fake_torch(monkeypatch, model)
    mono = np.linspace(-0.5, 0.5, 32)

    transformed, sr = rave_module.rave(
        mono,
        sr=8_000,
        model_path=exported_model_path,
    )

    assert sr == 8_000
    assert transformed.shape == (32, 2)
    np.testing.assert_allclose(transformed[:, 0], mono * 0.5, atol=1e-7)
    np.testing.assert_allclose(transformed[:, 1], mono * 0.5, atol=1e-7)
    assert model.last_input_shape == (1, 1, 32)


def test_rave_seed_is_applied_without_leaking_global_state(monkeypatch, exported_model_path):
    model = _FakeModel()
    _install_fake_torch(monkeypatch, model)

    rave_module.rave(
        np.zeros(32),
        sr=8_000,
        model_path=exported_model_path,
        seed=17,
    )

    assert _FakeDefaultGenerator.seeds == [17]


def test_rave_contract_applies_standard_rate_dividers():
    model = _FakeModel(
        sampling_rate=48_000,
        input_divider=2,
        output_divider=4,
    )

    contract = rave_module._model_contract(model)

    assert contract.input_sample_rate == 24_000
    assert contract.output_sample_rate == 12_000


@pytest.mark.parametrize(
    "forward_params",
    [
        [1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1.5, 1, 1],
    ],
)
def test_rave_contract_rejects_invalid_forward_metadata(forward_params):
    model = _FakeModel()
    model.forward_params = np.asarray(forward_params)

    with pytest.raises(ValueError, match="forward_params"):
        rave_module._model_contract(model)


def test_rave_contract_rejects_fractional_method_sample_rates():
    model = _FakeModel(sampling_rate=44_100, input_divider=8)

    with pytest.raises(ValueError, match="divisible"):
        rave_module._model_contract(model)


def test_asymmetric_model_rejects_ambiguous_stereo_input(monkeypatch, exported_model_path):
    model = _FakeModel(input_channels=1, output_channels=2)
    _install_fake_torch(monkeypatch, model)

    with pytest.raises(ValueError, match="expects 1 input channel.*produces 2"):
        rave_module.rave(
            np.zeros((32, 2)),
            sr=8_000,
            model_path=exported_model_path,
        )


def test_rave_limits_resampling_overshoot_before_model_inference(monkeypatch, exported_model_path):
    model = _FakeModel(sampling_rate=8_000)
    _install_fake_torch(monkeypatch, model)
    alternating = np.resize(np.array([-1.0, 1.0]), 64)

    rave_module.rave(alternating, sr=4_000, model_path=exported_model_path)

    assert model.last_input_peak <= 1.0


def test_rave_rejects_model_output_with_wrong_channel_count(monkeypatch, exported_model_path):
    model = _WrongChannelModel(input_channels=2)
    _install_fake_torch(monkeypatch, model)

    with pytest.raises(ValueError, match="batch/channel"):
        rave_module.rave(
            np.zeros((32, 2)),
            sr=8_000,
            model_path=exported_model_path,
        )


def test_rave_rejects_empty_output_sequence(monkeypatch, exported_model_path):
    model = _FakeModel()
    model.forward = lambda _tensor: ()
    _install_fake_torch(monkeypatch, model)

    with pytest.raises(ValueError, match="empty output sequence"):
        rave_module.rave(
            np.zeros(32),
            sr=8_000,
            model_path=exported_model_path,
        )


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_rave_model_integer_metadata_is_strict(value):
    model = _FakeModel()
    model.sampling_rate = value

    with pytest.raises(ValueError, match="positive 'sampling_rate'"):
        rave_module._model_int(model, "sampling_rate")


@pytest.mark.parametrize("seed", [-1, 1.5, True, "7"])
def test_rave_rejects_invalid_seed(monkeypatch, exported_model_path, seed):
    _install_fake_torch(monkeypatch, _FakeModel())

    with pytest.raises(ValueError, match="seed"):
        rave_module._validate_rave_parameters(exported_model_path, seed=seed)


def test_rave_output_parent_is_checked_before_model_loading(
    monkeypatch, tmp_path, exported_model_path
):
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
            model_path=exported_model_path,
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
    assert result.postprocess_params["seed"] == 0
    assert calls[0][2] is None
    assert calls[0][3]["device"] == "auto"
    assert calls[0][3]["seed"] == 0


def test_unavailable_rave_dependency_fails_before_primary_synthesis(monkeypatch, tmp_path):
    model_path = tmp_path / "style.ts"
    model_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        importlib.import_module("radiosonify.amplitude"),
        "amplitude_modulate",
        lambda *args, **kwargs: pytest.fail("primary synthesis must not run"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_preprocess_validated",
        lambda *args, **kwargs: pytest.fail("preprocessing must not run"),
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

    params = rave_module._validate_rave_parameters(model_path, "cuda:1")
    with pytest.raises(ValueError, match="detected 1 CUDA device"):
        rave_module._preflight_rave(
            input_channels=1,
            input_sample_rate=48_000,
            input_samples=48_000,
            **params,
        )


def test_unified_rave_contract_rejects_spatial_audio_before_preprocessing(monkeypatch, tmp_path):
    model_path = tmp_path / "asymmetric.ts"
    model_path.write_bytes(b"placeholder")
    _install_fake_torch(monkeypatch, _FakeModel(input_channels=1, output_channels=2))
    monkeypatch.setattr(
        pipeline_module,
        "_preprocess_validated",
        lambda *args, **kwargs: pytest.fail("preprocessing must not run"),
    )

    with pytest.raises(ValueError, match="input audio has 2"):
        rs.sonify(
            np.ones((2, 8, 8)),
            data_duration=0.01,
            method="spatial_erb",
            postprocess="rave",
            postprocess_params={"model_path": model_path},
        )


@pytest.mark.skipif(_real_torch is None, reason="requires the rave extra")
def test_real_torchscript_export_passes_preflight_and_inference(tmp_path):
    model_path = tmp_path / "tiny-rave.ts"
    scripted = _real_torch.jit.script(_TinyRaveExport())
    _real_torch.jit.save(scripted, model_path)
    params = rave_module._validate_rave_parameters(model_path, "cpu", 7)

    runtime = rave_module._preflight_rave(
        input_channels=1,
        input_sample_rate=8_000,
        input_samples=32,
        **params,
    )
    audio = np.linspace(-0.8, 0.8, 32, dtype=np.float32)
    transformed, sample_rate = rave_module.rave(
        audio,
        sr=8_000,
        **params,
        **runtime,
    )

    assert sample_rate == 8_000
    np.testing.assert_allclose(transformed, audio * 0.25, atol=1e-7)
