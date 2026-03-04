import numpy as np
import pytest
from astrosonify.core import (
    normalize,
    del_burst,
    rebin_spectrogram,
    to_profile,
    save_audio,
)


class TestNormalize:
    def test_output_range_0_to_1(self):
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = normalize(data)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_constant_array(self):
        data = np.ones(10) * 5.0
        result = normalize(data)
        assert np.all(result == 0.0)

    def test_2d_array(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = normalize(data)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)


class TestDelBurst:
    def test_output_range(self):
        rng = np.random.default_rng(42)
        data = rng.random((100, 50)) * 100 + 1
        result = del_burst(data, exposure_cut=25)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_shape_preserved(self):
        rng = np.random.default_rng(42)
        data = rng.random((100, 50)) * 100 + 1
        result = del_burst(data)
        assert result.shape == (100, 50)


class TestRebinSpectrogram:
    def test_downsample_both_axes(self):
        data = np.ones((100, 200))
        result = rebin_spectrogram(data, time_bins=50, freq_bins=100)
        assert result.shape == (50, 100)

    def test_none_keeps_original(self):
        data = np.ones((100, 200))
        result = rebin_spectrogram(data, time_bins=None, freq_bins=None)
        assert result.shape == (100, 200)

    def test_values_averaged(self):
        data = np.arange(12).reshape(4, 3).astype(float)
        result = rebin_spectrogram(data, time_bins=2, freq_bins=None)
        assert result.shape == (2, 3)
        np.testing.assert_array_almost_equal(result[0], [1.5, 2.5, 3.5])
        np.testing.assert_array_almost_equal(result[1], [7.5, 8.5, 9.5])

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2D"):
            rebin_spectrogram(np.ones(10), time_bins=5)


class TestToProfile:
    def test_2d_to_1d(self):
        data = np.ones((100, 50))
        result = to_profile(data)
        assert result.ndim == 1
        assert len(result) == 100

    def test_1d_passthrough(self):
        data = np.ones(100)
        result = to_profile(data)
        assert result.ndim == 1
        assert len(result) == 100

    def test_downsample(self):
        data = np.ones(100)
        result = to_profile(data, downsample=10)
        assert len(result) == 10


class TestSaveAudio:
    def test_writes_wav(self, tmp_path):
        audio = np.sin(np.linspace(0, 2 * np.pi, 48000)).astype(np.float32)
        path = tmp_path / "test.wav"
        save_audio(audio, 48000, str(path))
        assert path.exists()
        assert path.stat().st_size > 0
