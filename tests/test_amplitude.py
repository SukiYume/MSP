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

    def test_preprocessed_repeat_produces_equal_profile_cycles(self):
        """repeat 现在由预处理沿时间轴 tile；方法层只按输入长度循环插值。"""
        audio, _ = amplitude_modulate(
            np.tile(np.array([0.0, 1.0, 0.0, 0.0]), 5),
            sr=8_000,
            duration=1,
            freq=400,
        )

        cycles = audio.reshape(5, -1)
        for cycle in cycles[1:]:
            np.testing.assert_allclose(cycle, cycles[0], atol=2e-5)

    def test_2d_input(self):
        data = np.random.default_rng(42).random((100, 50))
        audio, sr = amplitude_modulate(data, sr=48000, duration=1)
        assert audio.ndim == 1

    def test_optional_compression_makes_weak_structure_audible(self):
        compressed = _compress_profile(np.array([0.0, 0.01, 1.0]), 99)

        assert compressed[0] == 0
        assert compressed[1] == pytest.approx(np.log1p(0.99) / np.log(100))
        assert compressed[1] > 0.1
        assert compressed[2] == pytest.approx(1)
        np.testing.assert_allclose(
            _compress_profile(np.array([0.0, 0.01, 1.0]), 0),
            [0.0, 0.01, 1.0],
        )

    def test_multiharmonic_carrier_adds_integer_partials(self):
        profile = np.array([0.0, 1.0, 1.0, 0.0])
        single, sr = amplitude_modulate(
            profile,
            sr=16_000,
            duration=1,
            freq=500,
            harmonics=1,
        )
        harmonic, _ = amplitude_modulate(
            profile,
            sr=sr,
            duration=1,
            freq=500,
            harmonics=4,
            harmonic_decay=1,
        )
        frequencies = np.fft.rfftfreq(len(single), 1 / sr)
        single_spectrum = np.abs(np.fft.rfft(single))
        harmonic_spectrum = np.abs(np.fft.rfft(harmonic))

        for partial in (2, 3, 4):
            index = int(np.argmin(np.abs(frequencies - 500 * partial)))
            assert harmonic_spectrum[index] > single_spectrum[index] * 100

    def test_partials_at_or_above_nyquist_are_omitted(self):
        audio, _ = amplitude_modulate(
            np.array([0.0, 1.0, 0.0]),
            sr=8_000,
            duration=0.1,
            freq=1_500,
            harmonics=10,
        )

        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) == pytest.approx(0.9)

    def test_extreme_harmonic_request_only_allocates_audible_partials(self):
        audio, _ = amplitude_modulate(
            np.array([0.0, 1.0, 0.0]),
            sr=8_000,
            duration=0.01,
            freq=3_000,
            harmonics=10**9,
        )

        assert audio.shape == (80,)
        assert np.all(np.isfinite(audio))

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

    def test_smallest_positive_frequency_is_handled_without_integer_overflow(self):
        audio, sr = amplitude_modulate(
            np.linspace(0, 1, 16),
            sr=8_000,
            duration=0.01,
            freq=np.nextafter(0.0, 1.0),
        )

        assert sr == 8_000
        assert len(audio) == 80
        assert np.all(np.isfinite(audio))

    def test_invalid_duration_raises(self):
        data = np.random.default_rng(42).random(100)
        with pytest.raises(ValueError, match="duration"):
            amplitude_modulate(data, duration=0)

    @pytest.mark.parametrize("compression", [-1, np.inf, True, "99"])
    def test_invalid_compression_raises(self, compression):
        with pytest.raises(ValueError, match="compression"):
            amplitude_modulate(np.arange(10.0), compression=compression)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"harmonics": 0}, "harmonics"),
            ({"harmonics": 1.5}, "harmonics"),
            ({"harmonic_decay": -1}, "harmonic_decay"),
            ({"harmonic_decay": np.inf}, "harmonic_decay"),
        ],
    )
    def test_invalid_harmonic_parameters_raise(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            amplitude_modulate(np.arange(10.0), **kwargs)

    def test_repeat_is_no_longer_a_method_parameter(self):
        """repeat 属于数据域，只能通过 preprocess/sonify 表达。"""
        with pytest.raises(TypeError, match="repeat"):
            amplitude_modulate(np.arange(10.0), repeat=5)

    def test_constant_profile_is_silence(self):
        audio, _ = amplitude_modulate(np.ones(100), duration=0.01)
        assert np.all(audio == 0)

    def test_rejects_profile_that_skipped_shared_preprocessing(self):
        with pytest.raises(ValueError, match="preprocess"):
            amplitude_modulate(np.arange(10.0), duration=0.01)
