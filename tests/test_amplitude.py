# tests/test_amplitude.py
import numpy as np
import pytest
from astrosonify.amplitude import amplitude_modulate


class TestAmplitudeModulate:
    def test_returns_tuple(self):
        data = np.random.default_rng(42).random(100)
        audio, sr = amplitude_modulate(data, sr=48000, duration=1)
        assert isinstance(audio, np.ndarray)
        assert sr == 48000

    def test_output_length(self):
        data = np.random.default_rng(42).random(100)
        audio, sr = amplitude_modulate(data, sr=48000, duration=2)
        assert len(audio) == 48000 * 2

    def test_2d_input(self):
        data = np.random.default_rng(42).random((100, 50))
        audio, sr = amplitude_modulate(data, sr=48000, duration=1)
        assert audio.ndim == 1

    def test_saves_to_file(self, tmp_path):
        data = np.random.default_rng(42).random(100)
        out = tmp_path / "out.wav"
        audio, sr = amplitude_modulate(data, sr=48000, duration=1, output=str(out))
        assert out.exists()

    def test_freq_zero_returns_finite_audio(self):
        data = np.random.default_rng(42).random(100)
        audio, _ = amplitude_modulate(data, sr=48000, duration=1, freq=0.0)
        assert np.all(np.isfinite(audio))

    def test_invalid_duration_raises(self):
        data = np.random.default_rng(42).random(100)
        with pytest.raises(ValueError, match="duration"):
            amplitude_modulate(data, duration=0)
