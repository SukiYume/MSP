import inspect

import numpy as np
import pytest

import radiosonify as rs
from radiosonify.array_ops import normalize
from radiosonify.preprocessing import _resize_axis


def test_public_preprocess_signature_matches_the_reported_defaults():
    signature = inspect.signature(rs.preprocess)

    assert {
        name: signature.parameters[name].default for name in rs.preprocessing_defaults()
    } == dict(rs.preprocessing_defaults())


def test_percentile_clipping_is_opt_in():
    data = np.arange(100, dtype=np.float64)

    default = rs.preprocess(data)
    explicit_off = rs.preprocess(data, clip_percentiles=None)
    clipped = rs.preprocess(data, clip_percentiles=(1, 99))

    assert rs.preprocessing_defaults()["clip_percentiles"] is None
    np.testing.assert_array_equal(default, explicit_off)
    assert not np.allclose(default, clipped)


@pytest.mark.parametrize("operation", [None, "none", " NONE "])
def test_baseline_correction_can_be_disabled(operation):
    data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    result = rs.preprocess(data, baseline_operation=operation)

    np.testing.assert_allclose(result, normalize(data))


def test_divide_mean_clip_and_normalize_matches_explicit_pipeline():
    data = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
    percentiles = (25.0, 75.0)

    result = rs.preprocess(
        data,
        baseline_operation="divide",
        baseline_statistic="mean",
        baseline_axis=0,
        clip_percentiles=percentiles,
    )

    divided = data / np.mean(data, axis=0, keepdims=True)
    lower, upper = np.percentile(divided, percentiles)
    expected = normalize(np.clip(divided, lower, upper))
    np.testing.assert_allclose(result, expected)


def test_preprocess_keeps_the_callers_array_unchanged():
    data = np.arange(1, 25, dtype=np.float64).reshape(6, 4)
    original = data.copy()

    result = rs.preprocess(
        data,
        baseline_operation="divide",
        baseline_statistic="mean",
        baseline_axis=0,
        clip_percentiles=(4, 96),
    )

    np.testing.assert_array_equal(data, original)
    assert not np.shares_memory(result, data)


def test_rebinning_happens_before_baseline_and_normalization():
    time = np.arange(64, dtype=np.float64)[:, None]
    feature = np.arange(16, dtype=np.float64)[None, :]
    data = 100 + 0.02 * time + 0.01 * feature
    data[24:40, 5:11] += np.linspace(0, 8, 16)[:, None]

    result = rs.preprocess(
        data,
        baseline_operation="divide",
        baseline_statistic="mean",
        baseline_axis=0,
        clip_percentiles=(1, 99),
        time_rebin=8,
        feature_rebin=4,
    )

    assert result.shape == (8, 4)
    assert result.min() == pytest.approx(0)
    assert result.max() == pytest.approx(1)
    assert np.std(result) > 0.1


def test_preprocess_can_upsample_time_and_feature_axes_generically():
    data = np.arange(12, dtype=np.float64).reshape(4, 3)

    result = rs.preprocess(data, time_rebin=10, feature_rebin=7)

    assert result.shape == (10, 7)
    assert np.all(np.isfinite(result))
    assert result.min() == pytest.approx(0)
    assert result.max() == pytest.approx(1)


def test_feature_resize_is_rejected_for_profiles():
    with pytest.raises(ValueError, match="feature_rebin"):
        rs.preprocess(np.arange(8.0), feature_rebin=4)


@pytest.mark.parametrize(
    ("shape", "data_type", "explicit_axis"),
    [
        ((8,), "profile", None),
        ((4, 3), "matrix", 0),
        ((2, 4, 3), "layered_matrix", 1),
    ],
)
def test_auto_baseline_axis_matches_standard_layout(shape, data_type, explicit_axis):
    data = np.arange(1, np.prod(shape) + 1, dtype=np.float64).reshape(shape)
    common = {
        "data_type": data_type,
        "baseline_operation": "divide",
        "baseline_statistic": "median",
        "clip_percentiles": (5, 95),
    }

    automatic = rs.preprocess(data, baseline_axis="auto", **common)
    explicit = rs.preprocess(data, baseline_axis=explicit_axis, **common)

    np.testing.assert_allclose(automatic, explicit)


def test_text_preprocessing_choices_are_case_and_whitespace_insensitive():
    result = rs.preprocess(
        np.arange(8.0),
        baseline_operation=" SUBTRACT ",
        baseline_statistic=" Median ",
        baseline_axis=" AUTO ",
        normalization_scope=" GLOBAL ",
    )

    assert result.min() == pytest.approx(0)
    assert result.max() == pytest.approx(1)


def test_mean_and_median_are_real_configurable_baseline_choices():
    data = np.array(
        [
            [1.0, 10.0],
            [1.0, 10.0],
            [1.0, 10.0],
            [100.0, 20.0],
        ]
    )
    common = {
        "baseline_operation": "divide",
        "baseline_axis": 0,
        "clip_percentiles": (1, 99),
    }

    mean = rs.preprocess(data, baseline_statistic="mean", **common)
    median = rs.preprocess(data, baseline_statistic="median", **common)

    assert not np.allclose(mean, median)
    assert mean.min() == pytest.approx(0)
    assert mean.max() == pytest.approx(1)
    assert median.min() == pytest.approx(0)
    assert median.max() == pytest.approx(1)


def test_near_zero_division_baseline_falls_back_without_inf():
    data = np.array([[1.0, 1.0], [-1.0, 2.0], [0.0, 3.0]])

    result = rs.preprocess(
        data,
        baseline_operation="divide",
        baseline_statistic="mean",
        baseline_axis=0,
    )

    assert np.all(np.isfinite(result))
    assert result.min() >= 0
    assert result.max() <= 1


def test_sparse_profile_is_not_erased_when_clip_bounds_coincide():
    profile = np.zeros(100)
    profile[37] = 10

    result = rs.preprocess(profile, clip_percentiles=(4, 96))

    assert np.count_nonzero(result) == 1
    assert result[37] == pytest.approx(1)


def test_constant_and_extreme_finite_inputs_remain_safe():
    np.testing.assert_array_equal(rs.preprocess(np.full(16, 7.0)), np.zeros(16))

    extreme = rs.preprocess(np.array([-1e308, 0.0, 1e308]))
    assert np.all(np.isfinite(extreme))
    assert extreme.min() == pytest.approx(0)
    assert extreme.max() == pytest.approx(1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"baseline_operation": "offset"}, "baseline_operation"),
        ({"baseline_statistic": "mode"}, "baseline_statistic"),
        ({"baseline_axis": 4}, "baseline_axis"),
        ({"clip_percentiles": (0, 99)}, "clip_percentiles"),
        ({"clip_percentiles": (99, 1)}, "clip_percentiles"),
        ({"nan_policy": "drop"}, "nan_policy"),
    ],
)
def test_invalid_preprocessing_settings_fail_clearly(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rs.preprocess(np.arange(8.0), **kwargs)


def test_preprocess_rejects_nonfinite_complex_and_wrong_type_shape():
    with pytest.raises(ValueError, match="nan_policy"):
        rs.preprocess(np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match="infinite"):
        rs.preprocess(np.array([0.0, np.inf]))
    with pytest.raises(ValueError, match="complex"):
        rs.preprocess(np.array([1 + 2j]))
    with pytest.raises(ValueError, match="requires a 2D"):
        rs.preprocess(np.ones(4), data_type="matrix")


# ---------- R1-4 逐通道噪声均衡 ----------


@pytest.mark.parametrize("statistic", ["std", "mad"])
def test_scale_statistic_equalizes_per_channel_noise(statistic):
    """把每个通道除以自身噪声尺度，是让爆发盖过带通/RFI 的关键一步。

    实测 FRB180301（rebin 到 2048x512）：只减中位数时通道噪声的 max/min 是
    17.3，加上这一步降到 1.8。这里用合成数据守住同一个性质。
    """
    rng = np.random.default_rng(0)
    channel_noise = np.geomspace(0.05, 5.0, 32)
    data = rng.normal(size=(512, 32)) * channel_noise

    without = rs.preprocess(data, clip_percentiles=None)
    with_scaling = rs.preprocess(data, scale_statistic=statistic, clip_percentiles=None)

    spread = lambda result: result.std(axis=0).max() / result.std(axis=0).min()  # noqa: E731
    assert spread(without) > 10
    assert spread(with_scaling) < 2


def test_mad_scaling_resists_a_single_blown_out_channel():
    """MAD 对被 RFI 打爆的通道更稳健，因此不会把其余通道压到听不见。"""
    rng = np.random.default_rng(1)
    data = rng.normal(size=(512, 16))
    data[:, 7] += rng.normal(size=512) * 50

    by_std = rs.preprocess(data, scale_statistic="std", clip_percentiles=None)
    by_mad = rs.preprocess(data, scale_statistic="mad", clip_percentiles=None)

    quiet_channels = [index for index in range(16) if index != 7]
    assert by_mad[:, quiet_channels].std() > by_std[:, quiet_channels].std()


def test_near_zero_scale_channels_are_left_unscaled_not_reoperated():
    """常量通道不缩放，而不是退化成另一种运算 —— 否则同一数组混用两种标定。"""
    data = np.zeros((64, 3))
    data[:, 0] = np.linspace(0, 1, 64)
    data[:, 1] = 5.0

    result = rs.preprocess(data, scale_statistic="std", clip_percentiles=None)

    assert np.all(np.isfinite(result))
    assert result[:, 1].std() == pytest.approx(0)


def test_scale_statistic_is_recorded_in_the_unified_result():
    result = rs.sonify(
        np.random.default_rng(2).normal(size=(32, 16)),
        data_duration=0.2,
        preprocess_params={"scale_statistic": "mad"},
    )

    assert result.preprocess_params["scale_statistic"] == "mad"


# ---------- R1-4 NaN 策略 ----------


def test_nan_is_rejected_by_default_and_handled_on_request():
    data = np.random.default_rng(3).normal(size=(32, 8))
    data[:, 3] = np.nan

    with pytest.raises(ValueError, match="nan_policy"):
        rs.preprocess(data)

    result = rs.preprocess(data, nan_policy="propagate")

    assert np.all(np.isfinite(result))
    # 掩掉的通道映射成静音，而不是凭空制造信号。
    assert np.all(result[:, 3] == 0)
    assert result[:, [0, 1, 2, 4]].std() > 0


def test_nan_aware_downsampling_renormalizes_valid_overlap():
    data = np.array([0.0, np.nan, 2.0, 4.0])

    result = _resize_axis(data, 2, axis=0, nan_aware=True)

    np.testing.assert_allclose(result, [0.0, 3.0])


def test_nan_aware_upsampling_does_not_spread_one_missing_sample():
    data = np.array([0.0, np.nan, 2.0])

    result = _resize_axis(data, 7, axis=0, nan_aware=True)

    assert np.count_nonzero(np.isnan(result)) == 1
    assert result[0] == pytest.approx(0)
    assert result[-1] == pytest.approx(2)


def test_masked_channels_do_not_distort_the_surviving_dynamic_range():
    """nan 感知的统计量必须忽略掩通道，否则量程会被 NaN 传染。"""
    rng = np.random.default_rng(4)
    clean = rng.normal(size=(64, 8))
    masked = clean.copy()
    masked[:, 5] = np.nan

    reference = rs.preprocess(clean[:, [0, 1, 2, 3, 4, 6, 7]], clip_percentiles=None)
    result = rs.preprocess(masked, nan_policy="propagate", clip_percentiles=None)

    np.testing.assert_allclose(
        result[:, [0, 1, 2, 3, 4, 6, 7]],
        reference,
        atol=1e-12,
    )


def test_infinite_values_are_rejected_under_every_nan_policy():
    data = np.ones((8, 4))
    data[0, 0] = np.inf

    for policy in ("raise", "propagate"):
        with pytest.raises(ValueError, match="infinite"):
            rs.preprocess(data, nan_policy=policy)


# ---------- R1-1 三维归一化范围 ----------


def test_layered_normalization_keeps_weak_layers_audible():
    """全局 min-max 会让弱层比强层安静几百倍，等于听不见。

    默认改成按层归一化后，每层都用满 [0, 1]；层与层真实的科学强度差应当由
    spatial 的 layer_gains 显式表达，而不是由归一化偶然决定。
    """
    rng = np.random.default_rng(5)
    strong = rng.gamma(4, 1, size=(64, 16)) * 10
    weak = rng.normal(0, 0.3, size=(64, 16))
    cube = np.stack([strong, weak, weak * 0.5, weak * 0.2])

    per_layer = rs.preprocess(cube)
    global_scope = rs.preprocess(cube, normalization_scope="global")

    per_layer_spread = per_layer.std(axis=(1, 2))
    global_spread = global_scope.std(axis=(1, 2))

    assert global_spread.max() / global_spread.min() > 100
    assert per_layer_spread.max() / per_layer_spread.min() < 3


def test_normalization_scope_defaults_by_dimensionality():
    assert (
        rs.sonify(
            np.random.default_rng(6).normal(size=(2, 32, 8)),
            data_duration=0.2,
        ).preprocess_params["normalization_scope"]
        == "per_layer"
    )
    assert (
        rs.sonify(
            np.random.default_rng(6).normal(size=(32, 8)),
            data_duration=0.2,
        ).preprocess_params["normalization_scope"]
        == "global"
    )


def test_per_layer_scope_is_rejected_for_lower_dimensional_data():
    with pytest.raises(ValueError, match="requires 3D"):
        rs.preprocess(np.arange(16.0).reshape(4, 4), normalization_scope="per_layer")


def test_layer_rebin_uses_ordered_area_averaging_before_normalization():
    data = np.arange(32.0).reshape(4, 4, 2)
    rebinned = rs.preprocess(
        data,
        layer_rebin=2,
        baseline_operation=None,
        clip_percentiles=None,
        normalization_scope="global",
    )
    expected = normalize(_resize_axis(data, 2, axis=0))

    assert rebinned.shape == (2, 4, 2)
    np.testing.assert_allclose(rebinned, expected)


@pytest.mark.parametrize(
    ("data", "layer_rebin", "message"),
    [
        (np.ones((4, 4)), 2, "only supported for 3D"),
        (np.ones((4, 4, 4)), "auto", "does not support 'auto'"),
        (np.ones((2, 4, 4)), 3, "cannot exceed input layer count"),
    ],
)
def test_layer_rebin_rejects_ambiguous_or_expansive_requests(data, layer_rebin, message):
    with pytest.raises(ValueError, match=message):
        rs.preprocess(data, layer_rebin=layer_rebin)


# ---------- R2-7 共享时间平滑 ----------


def test_shared_time_smoothing_reduces_isolated_time_domain_spikes():
    """这是第一部分里唯一的滤波步骤，之前没有任何直接测试。"""
    data = np.zeros((64, 8))
    data[32, :] = 1.0

    sharp = rs.preprocess(data, clip_percentiles=None)
    smoothed = rs.preprocess(data, time_smoothing=2.0, clip_percentiles=None)

    assert smoothed[31, 0] > sharp[31, 0]
    assert smoothed[32, 0] <= sharp[32, 0] + 1e-12
    assert smoothed.max() == pytest.approx(1)


def test_repeated_timeline_is_joined_before_time_smoothing():
    data = np.array([0.0, 0.25, 1.0, 0.5])

    repeated = rs.preprocess(
        data,
        repeat=2,
        time_smoothing=1.0,
        clip_percentiles=None,
    )
    explicit = rs.preprocess(
        np.tile(data, 2),
        time_smoothing=1.0,
        clip_percentiles=None,
    )

    np.testing.assert_allclose(repeated, explicit, atol=1e-12)


def test_time_smoothing_runs_along_time_for_layered_data():
    """三维布局的时间轴是轴 1；沿错轴平滑会把层与层混在一起。"""
    cube = np.zeros((3, 64, 8))
    cube[:, 32, :] = 1.0
    cube[1] *= 0.5

    smoothed = rs.preprocess(cube, time_smoothing=2.0, clip_percentiles=None)

    for layer in range(3):
        assert smoothed[layer, 31, 0] > 0
        assert smoothed[layer].max() == pytest.approx(1)


def test_masked_samples_survive_smoothing_without_spreading():
    """NaN 不能被高斯核抹到邻域，否则一个掩样本会污染整段时间。"""
    data = np.zeros((32, 4))
    data[:, 1] = np.nan
    data[16, 0] = 1.0

    result = rs.preprocess(
        data,
        nan_policy="propagate",
        time_smoothing=1.5,
        clip_percentiles=None,
    )

    assert np.all(np.isfinite(result))
    assert np.all(result[:, 1] == 0)
    assert result[16, 0] > result[14, 0]
