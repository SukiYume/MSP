# tests/test_profile.py
from unittest.mock import patch

import numpy as np
import pytest

from radiosonify.profile import profile_to_wave


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

    @patch("radiosonify.profile.get_instrument_path")
    def test_with_instrument(self, mock_get_path, tmp_path):
        import soundfile as sf

        phase = np.linspace(0, 2 * np.pi * 440, 2400)
        fake_wav = np.column_stack((np.sin(phase), 0.5 * np.sin(phase))).astype(np.float32)
        wav_path = tmp_path / "vio.wav"
        sf.write(str(wav_path), fake_wav, 24000)
        mock_get_path.return_value = str(wav_path)

        data = np.random.default_rng(42).random(100)
        audio, sr = profile_to_wave(data, sr=48000, duration=1, instrument="violin")
        assert len(audio) == 48000
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) <= 0.95 + 1e-6

    @patch("radiosonify.profile.get_instrument_path")
    def test_rejects_instrument_without_ac_signal(self, mock_get_path, tmp_path):
        import soundfile as sf

        wav_path = tmp_path / "silent.wav"
        sf.write(wav_path, np.ones(32, dtype=np.float32), 48000)
        mock_get_path.return_value = str(wav_path)

        with pytest.raises(ValueError, match="no usable AC signal"):
            profile_to_wave(np.arange(16.0), duration=0.01, instrument="violin")

    @patch("radiosonify.profile.get_instrument_path")
    def test_short_profile_keeps_causal_instrument_attack(self, mock_get_path, tmp_path):
        import soundfile as sf

        instrument = np.zeros(48_000, dtype=np.float32)
        instrument[:240] = np.sin(np.linspace(0, 8 * np.pi, 240))
        wav_path = tmp_path / "long-instrument.wav"
        sf.write(wav_path, instrument, 48_000)
        mock_get_path.return_value = str(wav_path)

        audio, _ = profile_to_wave(
            np.array([0.0, 1.0, 0.0, -0.5]),
            sr=48_000,
            duration=0.005,
            repeat=5,
            instrument="violin",
        )

        assert len(audio) == 240
        assert np.max(np.abs(audio)) > 0.1

    def test_saves_to_file(self, tmp_path):
        data = np.random.default_rng(42).random(100)
        out = tmp_path / "out.wav"
        audio, sr = profile_to_wave(data, sr=48000, duration=1, instrument=None, output=str(out))
        assert out.exists()

    def test_rejects_bad_output_path_before_reading_input_or_instrument(self, monkeypatch):
        monkeypatch.setattr(
            "radiosonify.profile.to_profile",
            lambda *args, **kwargs: pytest.fail("input preparation must not run"),
        )
        monkeypatch.setattr(
            "radiosonify.profile.get_instrument_path",
            lambda *args, **kwargs: pytest.fail("instrument preparation must not run"),
        )

        with pytest.raises(ValueError, match=".wav"):
            profile_to_wave(np.arange(8.0), output="bad.flac")

    def test_constant_profile_is_silence_not_dc(self):
        audio, _ = profile_to_wave(np.ones(100), sr=48000, duration=0.01, instrument=None)
        assert np.all(audio == 0)

    def test_rejects_unknown_instrument_even_for_constant_profile(self):
        with pytest.raises(ValueError, match="instrument"):
            profile_to_wave(np.ones(100), duration=0.01, instrument="drums")

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"sr": 0}, "sr"),
            ({"duration": 0}, "duration"),
            ({"repeat": 0}, "repeat"),
        ],
    )
    def test_rejects_invalid_output_parameters(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            profile_to_wave(np.arange(10.0), instrument=None, **kwargs)
