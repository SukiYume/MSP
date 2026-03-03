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
