# tests/test_profile.py
import numpy as np
import pytest
from unittest.mock import patch
from astrosonify.profile import profile_to_wave


class TestProfileToWave:
    def test_returns_tuple(self):
        data = np.random.default_rng(42).random(100)
        audio, sr = profile_to_wave(data, sr=48000, duration=1, instrument=None)
        assert isinstance(audio, np.ndarray)
        assert sr == 48000

    def test_output_length_matches_duration(self):
        data = np.random.default_rng(42).random(100)
        audio, sr = profile_to_wave(data, sr=48000, duration=2, instrument=None)
        assert len(audio) == 48000 * 2

    def test_2d_input_auto_averages(self):
        data = np.random.default_rng(42).random((100, 50))
        audio, sr = profile_to_wave(data, sr=48000, duration=1, instrument=None)
        assert audio.ndim == 1

    def test_no_instrument(self):
        data = np.random.default_rng(42).random(100)
        audio, sr = profile_to_wave(data, sr=48000, duration=1, instrument=None)
        assert len(audio) == 48000

    @patch("astrosonify.profile.get_instrument_path")
    def test_with_instrument(self, mock_get_path, tmp_path):
        import soundfile as sf
        fake_wav = np.sin(np.linspace(0, 2 * np.pi * 440, 4800)).astype(np.float32)
        wav_path = tmp_path / "vio.wav"
        sf.write(str(wav_path), fake_wav, 48000)
        mock_get_path.return_value = str(wav_path)

        data = np.random.default_rng(42).random(100)
        audio, sr = profile_to_wave(data, sr=48000, duration=1, instrument="violin")
        assert len(audio) == 48000

    def test_saves_to_file(self, tmp_path):
        data = np.random.default_rng(42).random(100)
        out = tmp_path / "out.wav"
        audio, sr = profile_to_wave(data, sr=48000, duration=1, instrument=None, output=str(out))
        assert out.exists()
