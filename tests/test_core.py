import builtins

import numpy as np
import pytest

from radiosonify.core import (
    _immutable_array,
    _interpolate_cyclic_profile,
    _merge_settings,
    del_burst,
    normalize,
    rebin_spectrogram,
    require,
    save_audio,
    to_profile,
)


def test_profile_bins_span_full_intervals_and_wrap_at_the_seam():
    """N 个 bin 覆盖 N 个等宽区间（分箱积分语义），末端接回第一个 bin。"""
    result = _interpolate_cyclic_profile(np.array([0.0, 1.0, 4.0]), n_samples=5)

    # 最后一个样本从 bin 2 插值回 bin 0，而不是停在 bin 2 上。
    np.testing.assert_allclose(result, [0.0, 0.6, 1.6, 3.4, 2.4])


def test_preprocessing_repeat_matches_the_removed_method_level_repeat():
    """repeat 上移到预处理后，结果必须与旧的方法内 repeat 逐样本一致。

    旧实现把 ``repeat`` 折进插值索引；新实现在预处理末尾沿时间轴 tile，然后
    以 repeat=1 循环插值。两者在数学上恒等，这条测试守住这个等价关系。
    """
    profile = np.array([0.0, 1.0, 4.0])
    tiled = np.tile(profile, 3)

    result = _interpolate_cyclic_profile(tiled, n_samples=9)

    np.testing.assert_allclose(result, [0.0, 1.0, 4.0, 0.0, 1.0, 4.0, 0.0, 1.0, 4.0])


def test_require_reports_broken_optional_binary(monkeypatch):
    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "torch":
            raise OSError("shm.dll could not be loaded")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)

    with pytest.raises(ImportError, match="installed but failed to load.*shm.dll"):
        require("torch", "hifigan")


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

    @pytest.mark.parametrize("data", [np.array([]), np.array([1.0, np.nan])])
    def test_rejects_empty_or_non_finite_input(self, data):
        with pytest.raises(ValueError):
            normalize(data)

    def test_extreme_finite_range_does_not_overflow(self):
        result = normalize(np.array([-1e308, 0.0, 1e308]))
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_complex_input_is_rejected_instead_of_losing_imaginary_part(self):
        with pytest.raises(ValueError, match="complex"):
            normalize(np.array([1 + 2j, 3 + 4j]))


@pytest.mark.filterwarnings("ignore:del_burst.*:DeprecationWarning")
class TestDelBurst:
    def test_public_helper_is_explicitly_deprecated(self):
        with pytest.warns(DeprecationWarning, match="preprocess"):
            del_burst(np.ones((4, 4)))

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

    def test_percentile_clip_matches_numpy_percentile(self):
        data = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
        exposure_cut = 4

        result = del_burst(data, exposure_cut=exposure_cut)

        col_mean = np.mean(data, axis=0)
        scaled = data / col_mean
        lower = np.percentile(scaled, 100.0 / exposure_cut)
        upper = np.percentile(scaled, 100.0 * (exposure_cut - 1) / exposure_cut)
        expected = normalize(np.clip(scaled, lower, upper))

        np.testing.assert_allclose(result, expected)

    def test_rejects_invalid_shape_and_exposure(self):
        with pytest.raises(ValueError, match="2D"):
            del_burst(np.ones(10))
        with pytest.raises(ValueError, match="exposure_cut"):
            del_burst(np.ones((10, 10)), exposure_cut=1)

    def test_near_zero_column_mean_does_not_create_infinite_values(self):
        data = np.array([[1.0, 1.0], [-1.0, 2.0], [1e-300, 3.0]])
        result = del_burst(data)
        assert np.all(np.isfinite(result))

    def test_extreme_finite_columns_do_not_overflow_during_cleaning(self):
        data = np.array([[1e308, 1e308], [1e308, -1e308]])
        result = del_burst(data)
        assert np.all(np.isfinite(result))


@pytest.mark.filterwarnings("ignore:rebin_spectrogram.*:DeprecationWarning")
class TestRebinSpectrogram:
    def test_public_helper_is_explicitly_deprecated(self):
        with pytest.warns(DeprecationWarning, match="preprocess"):
            rebin_spectrogram(np.ones((4, 4)), time_bins=2)

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

    def test_non_divisible_rebin_uses_the_full_axis(self):
        data = np.arange(10, dtype=np.float64).reshape(10, 1)
        result = rebin_spectrogram(data, time_bins=6)

        np.testing.assert_allclose(result[:, 0], [0.4, 2.0, 3.6, 5.4, 7.0, 8.6])
        assert float(np.mean(result)) == pytest.approx(float(np.mean(data)))

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2D"):
            rebin_spectrogram(np.ones(10), time_bins=5)

    def test_rejects_upsample_time(self):
        data = np.ones((10, 20))
        with pytest.raises(ValueError, match="time_bins"):
            rebin_spectrogram(data, time_bins=11)

    def test_rejects_upsample_freq(self):
        data = np.ones((10, 20))
        with pytest.raises(ValueError, match="freq_bins"):
            rebin_spectrogram(data, freq_bins=21)

    def test_rejects_non_positive_bins(self):
        data = np.ones((10, 20))
        with pytest.raises(ValueError, match="positive"):
            rebin_spectrogram(data, time_bins=0)
        with pytest.raises(ValueError, match="positive"):
            rebin_spectrogram(data, freq_bins=-1)

    def test_rejects_non_integer_bins(self):
        with pytest.raises(ValueError, match="positive integer"):
            rebin_spectrogram(np.ones((10, 20)), time_bins=2.5)


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

    def test_feature_axis_is_averaged_without_weighting(self):
        data = np.tile(np.array([0.0, 2.0, 4.0]), (5, 1))
        np.testing.assert_allclose(to_profile(data), np.full(5, 2.0))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            to_profile(np.array([]))


class TestSaveAudio:
    def test_writes_wav(self, tmp_path):
        import soundfile as sf

        audio = np.sin(np.linspace(0, 2 * np.pi, 48000)).astype(np.float32)
        path = tmp_path / "test.wav"
        save_audio(audio, 48000, str(path))
        assert path.exists()
        assert path.stat().st_size > 0
        assert sf.info(path).subtype == "PCM_16"

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "test.wav"
        save_audio(np.zeros(16, dtype=np.float32), 48000, path)
        assert path.exists()

    def test_writes_samples_by_channels_stereo(self, tmp_path):
        import soundfile as sf

        path = tmp_path / "stereo.wav"
        stereo = np.column_stack((np.linspace(-0.5, 0.5, 32), np.linspace(0.5, -0.5, 32)))
        save_audio(stereo, 8_000, path)

        data, sr = sf.read(path, always_2d=True)
        assert data.shape == (32, 2)
        assert sr == 8_000

    def test_rejects_clipping(self, tmp_path):
        with pytest.raises(ValueError, match="clipping"):
            save_audio(np.array([0.0, 1.1]), 48000, tmp_path / "bad.wav")

    def test_rejects_non_wav_path_before_writing(self, tmp_path):
        with pytest.raises(ValueError, match=r"\.wav"):
            save_audio(np.zeros(16), 48000, tmp_path / "bad.flac")
        assert not (tmp_path / "bad.flac").exists()


class TestImmutableArray:
    def test_strided_input_is_copied_exactly_once_and_stays_frozen(self):
        source = np.arange(24.0).reshape(2, 3, 4)
        transposed = np.transpose(source, (1, 2, 0))

        frozen = _immutable_array(transposed, dtype=np.float64)

        np.testing.assert_array_equal(frozen, transposed)
        assert frozen.flags.writeable is False
        # ``tobytes`` serializes the strided view directly, so the snapshot is
        # backed by an immutable bytes object without a separate contiguity pass.
        root = frozen
        while isinstance(root, np.ndarray):
            root = root.base
        assert isinstance(root, bytes)
        with pytest.raises(ValueError, match="WRITEABLE"):
            frozen.setflags(write=True)

        source[0, 0, 0] = 999.0
        assert frozen[0, 0, 0] == 0.0

    def test_object_arrays_are_rejected(self):
        with pytest.raises(TypeError, match="object arrays"):
            _immutable_array(np.array([object()], dtype=object))


class TestMergeSettings:
    defaults = {"alpha": 1, "beta": 2}

    def test_none_and_partial_mappings_fall_back_to_defaults(self):
        assert _merge_settings(self.defaults, None, field_name="f", unknown_label="u") == {
            "alpha": 1,
            "beta": 2,
        }
        assert _merge_settings(self.defaults, {"beta": 9}, field_name="f", unknown_label="u") == {
            "alpha": 1,
            "beta": 9,
        }

    def test_container_key_type_and_unknown_keys_are_reported_uniformly(self):
        with pytest.raises(ValueError, match="settings must be a mapping or None"):
            _merge_settings(self.defaults, [("alpha", 1)], field_name="settings", unknown_label="u")
        with pytest.raises(ValueError, match="settings keys must be strings"):
            _merge_settings(self.defaults, {1: "x"}, field_name="settings", unknown_label="u")
        with pytest.raises(ValueError, match="bad key: gamma; allowed: alpha, beta"):
            _merge_settings(
                self.defaults, {"gamma": 1}, field_name="settings", unknown_label="bad key"
            )

    def test_supplied_mapping_is_left_unchanged(self):
        supplied = {"beta": 9}
        _merge_settings(self.defaults, supplied, field_name="f", unknown_label="u")
        assert supplied == {"beta": 9}
