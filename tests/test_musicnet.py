import importlib
import inspect
import json
import pathlib
import sys
import types

import numpy as np
import pytest

import radiosonify as rs

musicnet_module = importlib.import_module("radiosonify.musicnet")


def test_rejects_bad_output_path_before_loading_audio_or_runtime(monkeypatch):
    monkeypatch.setattr(
        musicnet_module,
        "_load_audio_input",
        lambda *args, **kwargs: pytest.fail("audio loading must not run"),
    )

    with pytest.raises(ValueError, match=".wav"):
        musicnet_module.musicnet(np.zeros(800), output="bad.flac")


def test_legacy_batch_size_is_accepted_and_deprecated(monkeypatch):
    monkeypatch.setattr(
        musicnet_module,
        "_load_audio_input",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("validated")),
    )

    with pytest.warns(DeprecationWarning, match="batch_size"):
        with pytest.raises(RuntimeError, match="validated"):
            musicnet_module.musicnet(np.zeros(800), batch_size=4)


class _FakeCuda:
    @staticmethod
    def is_available():
        return False


class _FakeDevice:
    def __init__(self, name):
        self.type = name
        self.index = None


class _FakeContext:
    def __enter__(self):
        return self

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
        return _FakeContext()


class _FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    @property
    def shape(self):
        return self.data.shape

    def reshape(self, *shape):
        return _FakeTensor(self.data.reshape(*shape))

    def contiguous(self):
        return self

    def to(self, device):
        return self

    def size(self, dimension):
        return self.data.shape[dimension]

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class _FakeTorch:
    cuda = _FakeCuda()
    random = _FakeRandom()
    float32 = np.float32
    threads = 8

    @staticmethod
    def device(name):
        return _FakeDevice(name)

    @classmethod
    def get_num_threads(cls):
        return cls.threads

    @classmethod
    def set_num_threads(cls, value):
        cls.threads = value

    @staticmethod
    def inference_mode():
        return _FakeContext()

    @staticmethod
    def as_tensor(data, dtype=None, device=None):
        return _FakeTensor(np.asarray(data, dtype=dtype))

    @staticmethod
    def split(tensor, split_size, dimension):
        chunks = []
        for start in range(0, tensor.size(dimension), split_size):
            slices = [slice(None)] * tensor.data.ndim
            slices[dimension] = slice(start, start + split_size)
            chunks.append(_FakeTensor(tensor.data[tuple(slices)]))
        return tuple(chunks)

    @staticmethod
    def cat(tensors, dimension):
        return _FakeTensor(np.concatenate([tensor.data for tensor in tensors], axis=dimension))


class _FakeProgress:
    def __init__(self, total):
        self.total = total
        self.updates = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def update(self, amount):
        self.updates += amount


class _FakeTqdm:
    progresses = []

    @classmethod
    def tqdm(cls, *, total, desc, unit):
        assert desc == "Generating"
        assert unit == "step"
        progress = _FakeProgress(total)
        cls.progresses.append(progress)
        return progress


def test_checkpoint_loader_always_uses_weights_only():
    class FakeTorch:
        calls = []

        @classmethod
        def load(cls, path, **kwargs):
            cls.calls.append((path, kwargs))
            return {"encoder_state": {}, "decoder_state": {}}

    state = musicnet_module._load_checkpoint(FakeTorch, "checkpoint.pth")

    assert set(state) == {"encoder_state", "decoder_state"}
    assert FakeTorch.calls == [("checkpoint.pth", {"map_location": "cpu", "weights_only": True})]


def test_build_models_loads_both_states_and_constructs_native_rate_decoder(monkeypatch):
    instances = {}

    class FakeModel:
        def __init__(self, args):
            self.args = args
            self.state = None
            self.device = None

        def load_state_dict(self, state):
            self.state = state

        def eval(self):
            return self

        def to(self, device):
            self.device = device
            return self

    class FakeEncoder(FakeModel):
        def __init__(self, args):
            super().__init__(args)
            instances["encoder"] = self

    class FakeWaveNet(FakeModel):
        def __init__(self, args):
            super().__init__(args)
            instances["decoder"] = self

    class FakeGenerator:
        def __init__(self, decoder, *, wav_freq):
            self.decoder = decoder
            self.wav_freq = wav_freq

    fake_models = types.ModuleType("radiosonify.models.musicnet.wavenet_models")
    fake_models.Encoder = FakeEncoder
    fake_wavenet = types.ModuleType("radiosonify.models.musicnet.wavenet")
    fake_wavenet.WaveNet = FakeWaveNet
    fake_generator = types.ModuleType("radiosonify.models.musicnet.wavenet_generator")
    fake_generator.WavenetGenerator = FakeGenerator
    package = importlib.import_module("radiosonify.models.musicnet")

    monkeypatch.setattr(package, "wavenet_models", fake_models, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "radiosonify.models.musicnet.wavenet_models",
        fake_models,
    )
    monkeypatch.setitem(sys.modules, "radiosonify.models.musicnet.wavenet", fake_wavenet)
    monkeypatch.setitem(
        sys.modules,
        "radiosonify.models.musicnet.wavenet_generator",
        fake_generator,
    )
    monkeypatch.setattr(
        musicnet_module,
        "_load_checkpoint",
        lambda torch, path: {
            "encoder_state": {"encoder": 1},
            "decoder_state": {"decoder": 2},
        },
    )
    encoder_device = _FakeDevice("cuda")
    decoder_device = _FakeDevice("cpu")

    encoder, generator = musicnet_module._build_models(
        _FakeTorch,
        model_args=object(),
        checkpoint_path="checkpoint.pth",
        encoder_device=encoder_device,
        decoder_device=decoder_device,
        sr=16_000,
    )

    assert encoder is instances["encoder"]
    assert encoder.state == {"encoder": 1}
    assert encoder.device is encoder_device
    assert instances["decoder"].state == {"decoder": 2}
    assert instances["decoder"].device is decoder_device
    assert generator.decoder is instances["decoder"]
    assert generator.wav_freq == 16_000


def test_full_musicnet_orchestration_with_fake_runtime(monkeypatch, tmp_path):
    decoder_init_flags = []

    class FakeEncoder:
        def __call__(self, samples):
            assert samples.shape == (1, 1, 1_600)
            return _FakeTensor(np.ones((1, 4, 3), dtype=np.float32))

    class FakeDecoder:
        def generate(self, condition, *, init, pbar):
            decoder_init_flags.append(init)
            pbar.update(condition.size(2))
            values = np.full((1, 1, condition.size(2) * 4), 129, dtype=np.int16)
            return _FakeTensor(values)

    args_path = tmp_path / "args.json"
    args_path.write_text(json.dumps({"args": {"latent_d": 4}}), encoding="utf-8")
    checkpoint_path = tmp_path / "bestmodel_2.pth"
    checkpoint_path.write_bytes(b"fake")
    output_path = tmp_path / "styled.wav"
    _FakeTqdm.progresses.clear()
    _FakeDefaultGenerator.seeds.clear()

    monkeypatch.setattr(musicnet_module, "_require_torch", lambda: _FakeTorch)
    monkeypatch.setattr(musicnet_module, "_require_tqdm", lambda: _FakeTqdm)
    monkeypatch.setattr(
        musicnet_module,
        "get_model_path",
        lambda model, filename: str(args_path if filename == "args.json" else checkpoint_path),
    )
    monkeypatch.setattr(
        musicnet_module,
        "_build_models",
        lambda *args, **kwargs: (FakeEncoder(), FakeDecoder()),
    )
    audio, sr = musicnet_module.musicnet(
        np.linspace(-1, 1, 1_600),
        split_size=1,
        seed=7,
        output=output_path,
    )

    assert sr == 16_000
    assert len(audio) == 12
    assert audio.dtype == np.float32
    assert np.max(np.abs(audio)) == pytest.approx(0.95)
    assert decoder_init_flags == [True, False, False]
    assert _FakeDefaultGenerator.seeds == [7]
    assert _FakeTqdm.progresses[0].updates == 3
    assert _FakeTorch.threads == 8
    assert output_path.is_file()


def test_musicnet_defaults_to_the_pretrained_models_native_sample_rate():
    assert inspect.signature(musicnet_module.musicnet).parameters["sr"].default == 16_000
    assert rs.STYLE_NAMES is musicnet_module.STYLE_NAMES
    assert rs.STYLE_NAMES[2] == "Solo Piano (Bach)"


def test_array_input_is_resampled_to_the_models_native_sample_rate():
    source_sr = 48_000
    audio = np.sin(2 * np.pi * 440 * np.arange(4_800) / source_sr)

    prepared = musicnet_module._load_audio_input(audio, source_sr)

    assert len(prepared) == 1_600
    assert np.max(np.abs(prepared)) <= 1


def test_validation_does_not_make_a_redundant_float32_copy_before_mu_law():
    audio = np.linspace(-1, 1, 800, dtype=np.float64)

    prepared = musicnet_module._validate_audio_input(audio)

    assert prepared is audio
    assert prepared.dtype == np.float64


def test_musicnet_does_not_require_cuda_and_keeps_pathlib_intact(monkeypatch):
    original_posix = pathlib.PosixPath

    monkeypatch.setattr(musicnet_module, "_require_torch", lambda: _FakeTorch)
    monkeypatch.setattr(musicnet_module, "_require_tqdm", lambda: object())
    monkeypatch.setattr(
        musicnet_module,
        "get_model_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop-here")),
    )

    data = np.random.default_rng(42).random(800).astype(np.float32)
    with pytest.raises(RuntimeError, match="stop-here"):
        musicnet_module.musicnet(data)

    assert pathlib.PosixPath is original_posix


@pytest.mark.parametrize("decoder_id", [6, -1, 2.0, True])
def test_musicnet_decoder_id_validation(monkeypatch, decoder_id):
    monkeypatch.setattr(musicnet_module, "_require_torch", lambda: _FakeTorch)
    monkeypatch.setattr(musicnet_module, "_require_tqdm", lambda: object())

    with pytest.raises(ValueError, match="decoder_id"):
        musicnet_module.musicnet(np.zeros(800, dtype=np.float32), decoder_id=decoder_id)


def test_musicnet_rejects_too_short_input_before_loading_torch():
    with pytest.raises(ValueError, match="at least 800"):
        musicnet_module.musicnet(np.zeros(799, dtype=np.float32))


@pytest.mark.filterwarnings("ignore:aifc was removed in Python 3.13.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:sunau was removed in Python 3.13.*:DeprecationWarning")
def test_musicnet_rejects_short_wav_before_loading_torch(monkeypatch, tmp_path):
    import soundfile as sf

    wav_path = tmp_path / "short.wav"
    sf.write(wav_path, np.zeros(100, dtype=np.float32), 48000)
    monkeypatch.setattr(
        musicnet_module,
        "_require_torch",
        lambda: (_ for _ in ()).throw(AssertionError("torch should not load")),
    )

    with pytest.raises(ValueError, match="at least 800"):
        musicnet_module.musicnet(wav_path)


def test_temporary_num_threads_restores_after_error():
    class FakeTorch:
        threads = 8

        @classmethod
        def get_num_threads(cls):
            return cls.threads

        @classmethod
        def set_num_threads(cls, value):
            cls.threads = value

    with pytest.raises(RuntimeError, match="inference failed"):
        with musicnet_module._temporary_num_threads(FakeTorch, 1):
            assert FakeTorch.threads == 1
            raise RuntimeError("inference failed")

    assert FakeTorch.threads == 8


def test_musicnet_parameter_validation_rejects_bad_seed_and_checkpoint():
    defaults = {
        "decoder_id": 2,
        "checkpoint_type": "bestmodel",
        "split_size": 20,
        "num_threads": 1,
        "seed": 0,
    }
    with pytest.raises(ValueError, match="seed"):
        musicnet_module._validate_musicnet_parameters(**(defaults | {"seed": -1}))
    with pytest.raises(ValueError, match="checkpoint_type"):
        musicnet_module._validate_musicnet_parameters(
            **(defaults | {"checkpoint_type": ["bestmodel"]})
        )


def test_split_generation_only_resets_autoregressive_state_once():
    class FakeResult:
        def cpu(self):
            return self

    class FakeDecoder:
        def __init__(self):
            self.init_flags = []

        def generate(self, condition, *, init, pbar):
            self.init_flags.append(init)
            return FakeResult()

    decoder = FakeDecoder()
    generated = musicnet_module._generate_splits(
        decoder,
        [object(), object(), object()],
        pbar=object(),
    )

    assert len(generated) == 3
    assert decoder.init_flags == [True, False, False]
