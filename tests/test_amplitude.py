# tests/test_amplitude.py
import numpy as np
import pytest

from radiosonify.amplitude import _compress_profile, amplitude_modulate


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

    def test_repeat_produces_equal_profile_cycles_across_total_duration(self):
        audio, _ = amplitude_modulate(
            np.array([0.0, 1.0, 0.0, 0.0]),
            sr=8_000,
            duration=1,
            freq=400,
            repeat=5,
        )

        cycles = audio.reshape(5, -1)
        for cycle in cycles[1:]:
            np.testing.assert_allclose(cycle, cycles[0], atol=2e-5)

    def test_2d_input(self):
        data = np.random.default_rng(42).random((100, 50))
        audio, sr = amplitude_modulate(data, sr=48000, duration=1)
        assert audio.ndim == 1

    def test_default_compression_makes_weak_structure_audible(self):
        compressed = _compress_profile(np.array([0.0, 0.01, 1.0]), 99)

        assert compressed[0] == 0
        assert compressed[1] == pytest.approx(np.log1p(0.99) / np.log(100))
        assert compressed[1] > 0.1
        assert compressed[2] == pytest.approx(1)
        np.testing.assert_allclose(
            _compress_profile(np.array([0.0, 0.01, 1.0]), 0),
            [0.0, 0.01, 1.0],
        )

    def test_saves_to_file(self, tmp_path):
        data = np.random.default_rng(42).random(100)
        out = tmp_path / "out.wav"
        audio, sr = amplitude_modulate(data, sr=48000, duration=1, output=str(out))
        assert out.exists()

    def test_rejects_bad_output_path_before_reading_input(self, monkeypatch):
        monkeypatch.setattr(
            "radiosonify.amplitude.to_profile",
            lambda *args, **kwargs: pytest.fail("input preparation must not run"),
        )

        with pytest.raises(ValueError, match=".wav"):
            amplitude_modulate(np.arange(8.0), output="bad.flac")

    @pytest.mark.parametrize("freq", [0.0, -1.0, 24000.0, 49000.0, np.inf, True, "440"])
    def test_rejects_invalid_or_aliasing_frequency(self, freq):
        data = np.random.default_rng(42).random(100)
        with pytest.raises(ValueError, match="freq"):
            amplitude_modulate(data, sr=48000, duration=1, freq=freq)

    def test_invalid_duration_raises(self):
        data = np.random.default_rng(42).random(100)
        with pytest.raises(ValueError, match="duration"):
            amplitude_modulate(data, duration=0)

    @pytest.mark.parametrize("compression", [-1, np.inf, True, "99"])
    def test_invalid_compression_raises(self, compression):
        with pytest.raises(ValueError, match="compression"):
            amplitude_modulate(np.arange(10.0), compression=compression)

    @pytest.mark.parametrize("repeat", [0, -1, 1.5, True])
    def test_invalid_repeat_raises(self, repeat):
        with pytest.raises(ValueError, match="repeat"):
            amplitude_modulate(np.arange(10.0), repeat=repeat)

    def test_constant_profile_is_silence(self):
        audio, _ = amplitude_modulate(np.ones(100), duration=0.01)
        assert np.all(audio == 0)
