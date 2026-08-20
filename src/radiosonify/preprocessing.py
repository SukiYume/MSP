"""方法无关的科学数组预处理：全部数据域操作都发生在这里。

固定流水线顺序（每一步都可关闭，但顺序不可改）::

    层/时间/特征重分箱 -> 基线校正 -> 逐通道尺度归一 -> 分位裁剪 -> 重复 -> 时间平滑 -> 归一化

重分箱排在最前是实测结论，不是习惯。在真实 FRB 动态谱（FRB180301，
28346x4096 -> 2048x512）上对比"先分箱后校正"与"先校正后分箱"：

===================  ==================  =====================  =====
顺序                 通道噪声 max/min    爆发占满量程的比例     SNR
===================  ==================  =====================  =====
先分箱（本实现）     1.12                0.0606                 3.4
先校正后分箱         2.89                0.0135                 3.5
===================  ==================  =====================  =====

两者信噪比相同，但先分箱让爆发占据的输出动态范围大 4.5 倍 —— 因为在原始分辨率
上算分位数时，裁剪边界由单样本噪声起伏决定；分箱把噪声平均下去之后，信号就只
剩下 [0, 1] 里很窄的一段。响度对比直接取决于这个比例。
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from types import MappingProxyType
from typing import Any

import numpy as np

from .array_ops import (
    _MAD_TO_GAUSSIAN_SIGMA,
    _rebin_axis,
)
from .inputs import DataType, infer_data_type, parse_data_type
from .validation import (
    _as_real_array,
    _choice,
    _merge_settings,
    _positive_float,
    _positive_int,
)

_BASELINE_OPERATIONS = {"subtract", "divide"}
_BASELINE_STATISTICS = {"mean", "median"}
_SCALE_STATISTICS = {"std", "mad"}
_NORMALIZATION_SCOPES = {"auto", "global", "per_layer"}
_NAN_POLICIES = {"raise", "propagate"}

_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "time_rebin": None,
        "feature_rebin": None,
        "layer_rebin": None,
        "baseline_operation": "subtract",
        "baseline_statistic": "median",
        "baseline_axis": "auto",
        "scale_statistic": None,
        "clip_percentiles": None,
        "time_smoothing": None,
        "normalization_scope": "auto",
        "nan_policy": "raise",
    }
)

_NDIM_BY_TYPE = {
    DataType.PROFILE: 1,
    DataType.MATRIX: 2,
    DataType.LAYERED_MATRIX: 3,
}


def preprocessing_defaults() -> Mapping[str, Any]:
    """Return the immutable defaults used by the shared preprocessing stage."""
    return _DEFAULTS


# ---------- 参数校验 ----------


def _optional_choice(value: Any, *, name: str, choices: set[str]) -> str | None:
    """``None`` 表示关闭该步骤；其余值必须命中枚举。"""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return _choice(value, name=name, choices=choices)


def _automatic_baseline_axis(data_type: DataType) -> int | None:
    """返回标准布局中承载轮廓/扫描进度的轴。"""
    if data_type is DataType.PROFILE:
        return None
    if data_type is DataType.LAYERED_MATRIX:
        return 1
    return 0


def _resolve_axis(value: Any, *, data_type: DataType, ndim: int) -> int | None:
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            return _automatic_baseline_axis(data_type)
        raise ValueError("baseline_axis must be 'auto', None, or an integer axis")
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("baseline_axis must be 'auto', None, or an integer axis")
    axis = int(value)
    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        raise ValueError(f"baseline_axis must be between {-ndim} and {ndim - 1}")
    return axis


def _resolve_percentiles(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("clip_percentiles must contain two numbers between 0 and 100")
    try:
        raw = list(value)
    except TypeError as exc:
        raise ValueError("clip_percentiles must contain two numbers between 0 and 100") from exc
    if len(raw) != 2 or any(isinstance(item, (bool, np.bool_)) for item in raw):
        raise ValueError("clip_percentiles must contain two numbers between 0 and 100")
    try:
        lower, upper = (float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("clip_percentiles must contain two numbers between 0 and 100") from exc
    if not np.isfinite(lower) or not np.isfinite(upper) or not 0 < lower < upper < 100:
        raise ValueError("clip_percentiles must satisfy 0 < lower < upper < 100")
    return lower, upper


def _resolve_rebin(value: Any, *, name: str) -> int | str | None:
    """``None`` 保持原尺寸；``'auto'`` 交给方法几何解析；否则必须是正整数。"""
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            return "auto"
        raise ValueError(f"{name} must be 'auto', None, or a positive integer")
    return _positive_int(value, name=name)


def resolve_preprocess_params(
    data_type: DataType | str,
    supplied: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """合并并验证统一预处理参数，返回可记录的有效配置。"""
    resolved_type = parse_data_type(data_type)
    params = _merge_settings(
        _DEFAULTS,
        supplied,
        field_name="preprocess_params",
        unknown_label="unknown preprocessing parameter(s)",
    )
    ndim = _NDIM_BY_TYPE[resolved_type]

    params["time_rebin"] = _resolve_rebin(params["time_rebin"], name="time_rebin")
    params["feature_rebin"] = _resolve_rebin(params["feature_rebin"], name="feature_rebin")
    params["layer_rebin"] = _resolve_rebin(params["layer_rebin"], name="layer_rebin")
    if params["feature_rebin"] is not None and ndim == 1:
        raise ValueError("feature_rebin is only supported for 2D or 3D data")
    if params["layer_rebin"] == "auto":
        raise ValueError("layer_rebin does not support 'auto'; supply a target layer count")
    if params["layer_rebin"] is not None and ndim != 3:
        raise ValueError("layer_rebin is only supported for 3D layered data")
    params["baseline_operation"] = _optional_choice(
        params["baseline_operation"],
        name="baseline_operation",
        choices=_BASELINE_OPERATIONS,
    )
    params["baseline_statistic"] = _choice(
        params["baseline_statistic"],
        name="baseline_statistic",
        choices=_BASELINE_STATISTICS,
    )
    params["baseline_axis"] = _resolve_axis(
        params["baseline_axis"],
        data_type=resolved_type,
        ndim=ndim,
    )
    params["scale_statistic"] = _optional_choice(
        params["scale_statistic"],
        name="scale_statistic",
        choices=_SCALE_STATISTICS,
    )
    params["clip_percentiles"] = _resolve_percentiles(params["clip_percentiles"])
    if params["time_smoothing"] is not None:
        params["time_smoothing"] = _positive_float(
            params["time_smoothing"],
            name="time_smoothing",
        )
    scope = _choice(
        params["normalization_scope"],
        name="normalization_scope",
        choices=_NORMALIZATION_SCOPES,
    )
    if scope == "auto":
        # 三维默认按层归一化。全局 min-max 会让层间强度差直接决定可听性：
        # 实测模拟 IQUV 立方体时，弱层比强层安静 300 倍，等于听不见。
        # 层与层的科学相对强度应当通过 spatial 的 layer_gains 显式表达。
        scope = "per_layer" if resolved_type is DataType.LAYERED_MATRIX else "global"
    if scope == "per_layer" and ndim != 3:
        raise ValueError("normalization_scope='per_layer' requires 3D layered data")
    params["normalization_scope"] = scope
    params["nan_policy"] = _choice(
        params["nan_policy"],
        name="nan_policy",
        choices=_NAN_POLICIES,
    )
    return params


# ---------- 流水线各步骤 ----------


@contextmanager
def _quiet_all_nan_slices():
    """全部被掩掉的通道会让 nan* 归约发出 All-NaN 警告。

    这不是异常情况：这样的通道本来就没有可用数据，后面会由归一化后的填零
    处理成静音。把警告静音掉，避免用户以为出了问题。
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings(
            "ignore",
            message="Degrees of freedom <= 0",
            category=RuntimeWarning,
        )
        yield


def _reducer(statistic: str, *, nan_aware: bool):
    if statistic == "mean":
        return np.nanmean if nan_aware else np.mean
    return np.nanmedian if nan_aware else np.median


def _baseline_correct(
    data: np.ndarray,
    *,
    operation: str,
    statistic: str,
    axis: int | None,
    nan_aware: bool,
) -> np.ndarray:
    """在全局安全缩放域内做基线处理，避免有限极值运算溢出。"""
    # 始终只建立一个可写工作副本。大型动态谱可能有数 GiB；连续写出
    # ``scaled``、``scaled - baseline`` 等同尺寸临时数组会让本来可处理的
    # 输入因为峰值内存而失败。用极值而不是 ``np.abs(data)`` 求全局尺度，
    # 也避免再生成一个同尺寸绝对值数组。
    scaled = np.array(data, dtype=np.float64, copy=True)
    extremes = (np.nanmin, np.nanmax) if nan_aware else (np.min, np.max)
    data_min = float(extremes[0](scaled))
    data_max = float(extremes[1](scaled))
    global_scale = max(abs(data_min), abs(data_max))
    if global_scale != 0 and np.isfinite(global_scale):
        scaled /= global_scale
    baseline = _reducer(statistic, nan_aware=nan_aware)(scaled, axis=axis, keepdims=True)
    if operation == "subtract":
        np.subtract(scaled, baseline, out=scaled)
        return scaled

    # 除法用于校正乘性增益。零或近零基线通常表示正负抵消；对这些切片退化为
    # 减基线，既不制造 Inf，也不会把真实变化整列清零。
    threshold = np.sqrt(np.finfo(np.float64).eps)
    safe = np.abs(baseline) > threshold
    # ``where`` 会按轴广播。先只处理不安全切片的减法回退，再原位处理其余
    # 切片的除法，避免为结果再分配一个完整数组。
    np.subtract(scaled, baseline, out=scaled, where=~safe)
    np.divide(scaled, baseline, out=scaled, where=safe)
    return scaled


def _scale_normalize(
    data: np.ndarray,
    *,
    statistic: str,
    axis: int | None,
    nan_aware: bool,
) -> np.ndarray:
    """把每个通道除以自身的噪声尺度，使各通道的起伏幅度可比。

    这是让射电动态谱里的爆发盖过带通和 RFI 的关键一步。不做的话，归一化的
    量程会被噪声最大的少数通道占满：实测 FRB180301 上，只减中位数时通道噪声
    的 max/min 是 17.3，加上本步后降到 1.8。

    ``'mad'`` 用中位绝对偏差（乘 1.4826 换算成等价 sigma），比 ``'std'`` 抗
    强 RFI —— 一条被打爆的通道不会因为自身方差巨大而被过度压制到听不见。
    近零尺度的通道保持不缩放，而不是退化成另一种运算，以免同一个数组里
    混用两种互不可比的标定。
    """
    if statistic == "std":
        scale = (np.nanstd if nan_aware else np.std)(data, axis=axis, keepdims=True)
    else:
        median = (np.nanmedian if nan_aware else np.median)(data, axis=axis, keepdims=True)
        deviation = np.abs(data - median)
        scale = (np.nanmedian if nan_aware else np.median)(
            deviation,
            axis=axis,
            keepdims=True,
        )
        scale = scale * _MAD_TO_GAUSSIAN_SIGMA

    threshold = np.sqrt(np.finfo(np.float64).eps)
    safe = np.isfinite(scale) & (scale > threshold)
    np.divide(data, scale, out=data, where=safe)
    return data


def _resize_axis(
    data: np.ndarray,
    target_bins: int,
    *,
    axis: int,
    nan_aware: bool = False,
) -> np.ndarray:
    """Resize one scientific-data axis without discarding its endpoints.

    Downsampling uses exact equal-width area averaging. Upsampling linearly
    interpolates bin centres, which is preferable to duplicating samples and
    keeps this method-independent adapter usable for small images as well as
    large dynamic spectra.
    """
    target_bins = _positive_int(target_bins, name="target_bins")
    source_bins = data.shape[axis]
    if target_bins == source_bins:
        return data
    if target_bins < source_bins:
        return _rebin_axis(data, target_bins, axis=axis, nan_aware=nan_aware)

    moved = np.moveaxis(data, axis, 0)
    positions = (np.arange(target_bins, dtype=np.float64) + 0.5) * (source_bins / target_bins) - 0.5
    positions = np.clip(positions, 0.0, source_bins - 1.0)
    lower = np.floor(positions).astype(np.intp)
    upper = np.minimum(lower + 1, source_bins - 1)
    fraction = positions - lower
    fraction = fraction.reshape((target_bins,) + (1,) * (moved.ndim - 1))
    lower_values = moved[lower]
    upper_values = moved[upper]
    if nan_aware:
        lower_weight = 1.0 - fraction
        upper_weight = fraction
        lower_valid = ~np.isnan(lower_values)
        upper_valid = ~np.isnan(upper_values)
        numerator = (
            np.where(lower_valid, lower_values, 0.0) * lower_weight
            + np.where(upper_valid, upper_values, 0.0) * upper_weight
        )
        denominator = lower_valid * lower_weight + upper_valid * upper_weight
        resized = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )
    else:
        resized = lower_values * (1.0 - fraction) + upper_values * fraction
    return np.moveaxis(resized, 0, axis)


def _minmax_owned(data: np.ndarray) -> np.ndarray:
    """Min-max normalize an owned working array in place when possible."""
    data_min = float(np.nanmin(data))
    data_max = float(np.nanmax(data))
    if not np.isfinite(data_min) or not np.isfinite(data_max) or data_max == data_min:
        data.fill(0.0)
        return data

    scale = max(abs(data_min), abs(data_max))
    data /= scale
    scaled_min = data_min / scale
    scaled_max = data_max / scale
    data -= scaled_min
    data /= scaled_max - scaled_min
    # Only repair tiny floating-point excursions; the mapping stays linear.
    np.clip(data, 0.0, 1.0, out=data)
    return data


def _normalize_owned(data: np.ndarray, *, scope: str) -> np.ndarray:
    if scope == "global":
        return _minmax_owned(data)
    for layer in range(data.shape[0]):
        _minmax_owned(data[layer])
    return data


def _tile_time_axis(
    data: np.ndarray,
    repeat: int,
    *,
    time_axis: int,
    overlap: int = 0,
) -> np.ndarray:
    """沿时间轴重复数据，并允许相邻副本共享固定数量的边界帧。"""
    if repeat == 1:
        return data
    if overlap < 0 or overlap >= data.shape[time_axis]:
        raise ValueError("repeat overlap must be smaller than the time-axis length")
    if overlap:
        tail = [slice(None)] * data.ndim
        tail[time_axis] = slice(overlap, None)
        return np.concatenate([data] + [data[tuple(tail)]] * (repeat - 1), axis=time_axis)
    reps = [1] * data.ndim
    reps[time_axis] = repeat
    return np.tile(data, reps)


def _time_axis_for(ndim: int) -> int:
    """标准布局下的时间轴：1-D/2-D 是轴 0，3-D ``(layer, time, feature)`` 是轴 1。"""
    return 0 if ndim < 3 else 1


def _preprocess_validated(
    data: np.ndarray,
    params: Mapping[str, Any],
    *,
    repeat: int = 1,
    repeat_overlap: int = 0,
) -> np.ndarray:
    """Run the common preprocessing pipeline on an already validated array.

    See the module docstring for why the fixed order starts with rebinning.
    Repeated copies are joined before temporal smoothing so the Gaussian filter
    also covers internal boundaries. Without smoothing, tiling remains a final
    memory-saving operation because min-max normalization commutes with copies.
    """
    nan_aware = params["nan_policy"] == "propagate"
    if not nan_aware and np.isnan(data).any():
        # 输入契约放行 NaN（掩通道是真实存在的），是否接受由这里的策略决定，
        # 这样 sonify() 的默认行为仍然是"含 NaN 就报错"。
        raise ValueError(
            "data contains NaN; set preprocess_params={'nan_policy': 'propagate'} "
            "to treat them as masked samples"
        )
    quiet = _quiet_all_nan_slices() if nan_aware else nullcontext()
    with quiet:
        return _run_pipeline(
            data,
            params,
            repeat=repeat,
            repeat_overlap=repeat_overlap,
            nan_aware=nan_aware,
        )


def _resize_scientific_axes(
    data: np.ndarray,
    params: Mapping[str, Any],
    *,
    nan_aware: bool,
) -> np.ndarray:
    """Apply explicit layer, time, and feature geometry in canonical order."""
    time_axis = _time_axis_for(data.ndim)
    feature_axis = None if data.ndim == 1 else data.ndim - 1
    working = data
    if params["layer_rebin"] is not None:
        target_layers = params["layer_rebin"]
        if target_layers > working.shape[0]:
            raise ValueError(
                f"layer_rebin ({target_layers}) cannot exceed input layer count ({working.shape[0]})"
            )
        working = _rebin_axis(working, target_layers, axis=0, nan_aware=nan_aware)
    if params["time_rebin"] is not None:
        working = _resize_axis(
            working,
            params["time_rebin"],
            axis=time_axis,
            nan_aware=nan_aware,
        )
    if params["feature_rebin"] is not None:
        if feature_axis is None:
            raise ValueError("feature_rebin is only supported for 2D or 3D data")
        working = _resize_axis(
            working,
            params["feature_rebin"],
            axis=feature_axis,
            nan_aware=nan_aware,
        )
    return working


def _calibrate_array(
    data: np.ndarray,
    params: Mapping[str, Any],
    *,
    nan_aware: bool,
) -> np.ndarray:
    """Apply baseline and per-channel scale calibration."""
    working = data

    if params["baseline_operation"] is not None:
        working = _baseline_correct(
            working,
            operation=params["baseline_operation"],
            statistic=params["baseline_statistic"],
            axis=params["baseline_axis"],
            nan_aware=nan_aware,
        )
    if params["scale_statistic"] is not None:
        working = _scale_normalize(
            working,
            statistic=params["scale_statistic"],
            axis=params["baseline_axis"],
            nan_aware=nan_aware,
        )
    return working


def _clip_array(
    data: np.ndarray,
    percentiles: tuple[float, float] | None,
    *,
    nan_aware: bool,
) -> np.ndarray:
    """Apply a global percentile interval when it has non-zero width."""
    if percentiles is None:
        return data
    percentile = np.nanpercentile if nan_aware else np.percentile
    lower, upper = percentile(data, percentiles)
    if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
        np.clip(data, lower, upper, out=data)
    return data


def _smooth_time_axis(
    data: np.ndarray,
    sigma: float | None,
    *,
    time_axis: int,
    nan_aware: bool,
) -> np.ndarray:
    """Smooth the canonical time axis while retaining a propagated NaN mask."""
    if sigma is None:
        return data
    from scipy.ndimage import gaussian_filter1d

    if not nan_aware:
        return gaussian_filter1d(data, sigma=sigma, axis=time_axis, mode="reflect")
    mask = np.isnan(data)
    working = data
    if mask.any():
        filler = np.nanmedian(working, axis=time_axis, keepdims=True)
        working = np.where(mask, np.nan_to_num(filler), working)
    working = gaussian_filter1d(working, sigma=sigma, axis=time_axis, mode="reflect")
    working[mask] = np.nan
    return working


def _run_pipeline(
    data: np.ndarray,
    params: Mapping[str, Any],
    *,
    repeat: int,
    repeat_overlap: int,
    nan_aware: bool,
) -> np.ndarray:
    time_axis = _time_axis_for(data.ndim)
    working = np.array(data, dtype=np.float64, copy=True)
    working = _resize_scientific_axes(working, params, nan_aware=nan_aware)
    working = _calibrate_array(working, params, nan_aware=nan_aware)
    working = _clip_array(
        working,
        params["clip_percentiles"],
        nan_aware=nan_aware,
    )

    tiled_before_smoothing = params["time_smoothing"] is not None and repeat > 1
    if tiled_before_smoothing:
        working = _tile_time_axis(
            working,
            repeat,
            time_axis=time_axis,
            overlap=repeat_overlap,
        )

    working = _smooth_time_axis(
        working,
        params["time_smoothing"],
        time_axis=time_axis,
        nan_aware=nan_aware,
    )

    # ``working`` is owned by this function, including every resize result.
    # Public ``preprocess`` therefore continues not to modify caller input.
    working = _normalize_owned(working, scope=params["normalization_scope"])
    if nan_aware:
        # 掩掉的样本在 [0, 1] 里没有"正确"取值。填 0 表示静音，是唯一不会
        # 凭空制造信号的选择；这必须在归一化之后做，否则 0 会参与算量程。
        np.nan_to_num(working, copy=False, nan=0.0)
    if tiled_before_smoothing:
        return working
    return _tile_time_axis(
        working,
        repeat,
        time_axis=time_axis,
        overlap=repeat_overlap,
    )


def preprocess(
    data: np.ndarray,
    *,
    data_type: DataType | str | None = None,
    time_rebin: int | None = _DEFAULTS["time_rebin"],
    feature_rebin: int | None = _DEFAULTS["feature_rebin"],
    layer_rebin: int | None = _DEFAULTS["layer_rebin"],
    repeat: int = 1,
    baseline_operation: str | None = _DEFAULTS["baseline_operation"],
    baseline_statistic: str = _DEFAULTS["baseline_statistic"],
    baseline_axis: str | int | None = _DEFAULTS["baseline_axis"],
    scale_statistic: str | None = _DEFAULTS["scale_statistic"],
    clip_percentiles: Sequence[float] | None = _DEFAULTS["clip_percentiles"],
    time_smoothing: float | None = _DEFAULTS["time_smoothing"],
    normalization_scope: str = _DEFAULTS["normalization_scope"],
    nan_policy: str = _DEFAULTS["nan_policy"],
) -> np.ndarray:
    """在任何声化方法之前把 1-D/2-D/3-D 科学数组映射到 ``[0, 1]``。

    这是框架的第一部分，也是唯一一处允许改动数据的地方。处理顺序固定为
    重分箱、基线校正、逐通道尺度归一、分位裁剪、重复、时间平滑、归一化；
    模块文档解释了为什么重分箱必须排在最前。

    Args:
        data: 1-D 轮廓、2-D 矩阵或 3-D ``(layer, time, feature)`` 分层矩阵。
        data_type: 显式类型，``None`` 表示按维数推断。
        time_rebin: 时间轴目标格数。缩小用等宽面积平均，放大用 bin 中心线性
            插值，都不要求整除。``None`` 保持原尺寸。
        feature_rebin: 特征轴目标格数，仅 2-D/3-D 可用。
        layer_rebin: 三维数据的目标层数，使用有序面积平均降维；``None`` 保持层数。
        repeat: 沿时间轴重复数据的遍数。
        baseline_operation: ``'subtract'`` 校正加性基线，``'divide'`` 校正乘性增益，
            ``None`` 关闭基线校正。
        baseline_statistic: 基线统计量 ``'mean'`` 或 ``'median'``。
        baseline_axis: ``'auto'`` 对 profile 用全局统计，对二维沿轴 0 逐列，
            对 ``(layer, time, feature)`` 沿轴 1；也可给显式整数轴或 ``None``。
            该轴同时决定 ``scale_statistic`` 的逐通道方向：两步都在同一组
            通道上定标，才能得到彼此可比的结果。
        scale_statistic: ``'std'`` 或 ``'mad'`` 时把每个通道除以自身噪声尺度，
            使各通道起伏可比；``None`` 关闭。射电动态谱建议用 ``'mad'``。
            通道方向取自 ``baseline_axis``。
        clip_percentiles: 双侧分位裁剪区间，``None`` 关闭。分位点在整个数组上
            统计，包括 3-D 的全部层；``normalization_scope`` 只决定随后
            min-max 的作用范围。
        time_smoothing: 沿时间轴的高斯 sigma（以格为单位），``None`` 关闭。
        normalization_scope: ``'global'`` 全数组统一量程；``'per_layer'`` 每层
            独立，仅 3-D 可用；``'auto'`` 对 3-D 选 per_layer，其余选 global。
        nan_policy: ``'raise'`` 拒绝含 NaN 的输入；``'propagate'`` 用 nan 感知的
            统计量处理掩通道，并在归一化后把它们填 0。``Inf`` 始终拒绝。
    """
    # 始终在输入层放行 NaN，由下面已解析的 nan_policy 统一决定接受与否；
    # 这样 'raise' 报的是"改 nan_policy 就能处理"，而不是笼统的"必须有限"。
    array = _as_real_array(data, name="data", ndim=(1, 2, 3), allow_nan=True)
    resolved_type = infer_data_type(array) if data_type is None else parse_data_type(data_type)
    expected_ndim = _NDIM_BY_TYPE[resolved_type]
    if array.ndim != expected_ndim:
        raise ValueError(
            f"data_type='{resolved_type.value}' requires a {expected_ndim}D array, "
            f"got {array.ndim}D"
        )
    params = resolve_preprocess_params(
        resolved_type,
        {
            "time_rebin": time_rebin,
            "feature_rebin": feature_rebin,
            "layer_rebin": layer_rebin,
            "baseline_operation": baseline_operation,
            "baseline_statistic": baseline_statistic,
            "baseline_axis": baseline_axis,
            "scale_statistic": scale_statistic,
            "clip_percentiles": clip_percentiles,
            "time_smoothing": time_smoothing,
            "normalization_scope": normalization_scope,
            "nan_policy": nan_policy,
        },
    )
    if params["time_rebin"] == "auto":
        raise ValueError("time_rebin='auto' requires a sonification method; use sonify()")
    return _preprocess_validated(
        array,
        params,
        repeat=_positive_int(repeat, name="repeat"),
    )


def _as_normalized_array(
    data: Any,
    *,
    name: str = "data",
    ndim: int | tuple[int, ...] | None = None,
) -> np.ndarray:
    """验证低层方法收到统一预处理后的 ``[0, 1]`` 输入，不修改数值。"""
    array = _as_real_array(data, name=name, ndim=ndim, allow_nan=False)
    if float(np.min(array)) < 0 or float(np.max(array)) > 1:
        raise ValueError(
            f"{name} must be normalized to [0, 1] before sonification; "
            "call radiosonify.preprocess() or use radiosonify.sonify()"
        )
    return array


__all__ = ["preprocess", "preprocessing_defaults", "resolve_preprocess_params"]
