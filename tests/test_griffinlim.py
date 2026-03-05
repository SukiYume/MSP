import numpy as np
import pytest
from radiosonify.griffinlim import griffinlim


LEGACY_GRIFFIN_TIME_BINS = 128
LEGACY_GRIFFIN_FREQ_BINS = 512


class TestGriffinLim:
    def test_returns_tuple(self):
        rng = np.random.default_rng(42)
        spec = rng.random((256, 1024))
        audio, sr = griffinlim(
            spec,
            sr=48000,
            n_iter=10,
            n_mels=LEGACY_GRIFFIN_FREQ_BINS,
            time_rebin=LEGACY_GRIFFIN_TIME_BINS,
            freq_rebin=LEGACY_GRIFFIN_FREQ_BINS,
        )
        assert isinstance(audio, np.ndarray)
        assert sr == 48000

    def test_output_is_1d(self):
        rng = np.random.default_rng(42)
        spec = rng.random((256, 1024))
        audio, sr = griffinlim(
            spec,
            sr=48000,
            n_iter=10,
            n_mels=LEGACY_GRIFFIN_FREQ_BINS,
            time_rebin=LEGACY_GRIFFIN_TIME_BINS,
            freq_rebin=LEGACY_GRIFFIN_FREQ_BINS,
        )
        assert audio.ndim == 1
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 1e-6

    def test_rejects_1d(self):
        with pytest.raises(ValueError):
            griffinlim(np.ones(100), sr=48000)

    def test_auto_rebin_freq(self):
        rng = np.random.default_rng(42)
        spec = rng.random((256, 1024))
        audio, sr = griffinlim(
            spec,
            sr=48000,
            n_iter=10,
            n_mels=LEGACY_GRIFFIN_FREQ_BINS,
            time_rebin=LEGACY_GRIFFIN_TIME_BINS,
            freq_rebin=LEGACY_GRIFFIN_FREQ_BINS,
        )
        assert audio.ndim == 1

    def test_saves_to_file(self, tmp_path):
        rng = np.random.default_rng(42)
        spec = rng.random((256, 1024))
        out = tmp_path / "out.wav"
        audio, sr = griffinlim(
            spec,
            sr=48000,
            n_iter=10,
            n_mels=LEGACY_GRIFFIN_FREQ_BINS,
            time_rebin=LEGACY_GRIFFIN_TIME_BINS,
            freq_rebin=LEGACY_GRIFFIN_FREQ_BINS,
            output=str(out),
        )
        assert out.exists()
