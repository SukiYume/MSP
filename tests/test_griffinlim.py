import numpy as np
import pytest
from astrosonify.griffinlim import griffinlim


class TestGriffinLim:
    def test_returns_tuple(self):
        rng = np.random.default_rng(42)
        spec = rng.random((64, 128))
        audio, sr = griffinlim(spec, sr=48000, n_iter=10, n_mels=128)
        assert isinstance(audio, np.ndarray)
        assert sr == 48000

    def test_output_is_1d(self):
        rng = np.random.default_rng(42)
        spec = rng.random((64, 128))
        audio, sr = griffinlim(spec, sr=48000, n_iter=10, n_mels=128)
        assert audio.ndim == 1

    def test_rejects_1d(self):
        with pytest.raises(ValueError):
            griffinlim(np.ones(100), sr=48000)

    def test_auto_rebin_freq(self):
        rng = np.random.default_rng(42)
        spec = rng.random((64, 256))
        audio, sr = griffinlim(spec, sr=48000, n_iter=10, n_mels=128)
        assert audio.ndim == 1

    def test_saves_to_file(self, tmp_path):
        rng = np.random.default_rng(42)
        spec = rng.random((32, 64))
        out = tmp_path / "out.wav"
        audio, sr = griffinlim(spec, sr=48000, n_iter=10, n_mels=64, output=str(out))
        assert out.exists()
