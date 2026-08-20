import numpy as np
import pytest

from radiosonify.array_ops import (
    _interpolate_cyclic_profile,
    _rebin_axis,
    normalize,
    to_profile,
)


def test_profile_bins_span_full_intervals_and_wrap_at_the_seam():
    result = _interpolate_cyclic_profile(np.array([0.0, 1.0, 4.0]), n_samples=5)
    np.testing.assert_allclose(result, [0.0, 0.6, 1.6, 3.4, 2.4])


def test_preprocessed_repetition_interpolates_without_new_seams():
    tiled = np.tile(np.array([0.0, 1.0, 4.0]), 3)
    result = _interpolate_cyclic_profile(tiled, n_samples=9)
    np.testing.assert_allclose(result, [0.0, 1.0, 4.0, 0.0, 1.0, 4.0, 0.0, 1.0, 4.0])


class TestNormalize:
    def test_output_range_0_to_1(self):
        result = normalize(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_constant_array(self):
        assert np.all(normalize(np.ones(10) * 5.0) == 0.0)

    def test_2d_array(self):
        result = normalize(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    @pytest.mark.parametrize("data", [np.array([]), np.array([1.0, np.nan])])
    def test_rejects_empty_or_non_finite_input(self, data):
        with pytest.raises(ValueError):
            normalize(data)

    def test_extreme_finite_range_does_not_overflow(self):
        np.testing.assert_allclose(normalize(np.array([-1e308, 0.0, 1e308])), [0.0, 0.5, 1.0])

    def test_complex_input_is_rejected(self):
        with pytest.raises(ValueError, match="complex"):
            normalize(np.array([1 + 2j, 3 + 4j]))


class TestRebinAxis:
    def test_downsamples_both_axes_by_area(self):
        data = np.ones((100, 200))
        result = _rebin_axis(_rebin_axis(data, 50, axis=0), 100, axis=1)
        assert result.shape == (50, 100)

    def test_values_are_averaged(self):
        data = np.arange(12).reshape(4, 3).astype(float)
        result = _rebin_axis(data, 2, axis=0)
        np.testing.assert_array_almost_equal(result, [[1.5, 2.5, 3.5], [7.5, 8.5, 9.5]])

    def test_non_divisible_rebin_uses_full_axis(self):
        data = np.arange(10, dtype=np.float64).reshape(10, 1)
        result = _rebin_axis(data, 6, axis=0)
        np.testing.assert_allclose(result[:, 0], [0.4, 2.0, 3.6, 5.4, 7.0, 8.6])
        assert float(np.mean(result)) == pytest.approx(float(np.mean(data)))


class TestToProfile:
    def test_2d_to_1d(self):
        result = to_profile(np.ones((100, 50)))
        assert result.shape == (100,)

    def test_1d_passthrough(self):
        result = to_profile(np.ones(100))
        assert result.shape == (100,)

    def test_feature_axis_is_averaged_without_weighting(self):
        data = np.tile(np.array([0.0, 2.0, 4.0]), (5, 1))
        np.testing.assert_allclose(to_profile(data), np.full(5, 2.0))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            to_profile(np.array([]))
