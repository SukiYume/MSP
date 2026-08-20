"""RadioSonify 的公共数值校验、数据变换与 WAV I/O 工具。"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

_MAD_TO_GAUSSIAN_SIGMA = 1.4826

# ---------- 数值输入与参数校验 ----------


def _as_real_array(
    data: Any,
    *,
    name: str = "data",
    ndim: int | tuple[int, ...] | None = None,
    allow_nan: bool = False,
) -> np.ndarray:
    """把输入转换为非空实数 ``float64`` 数组。

    ``allow_nan=True`` 只放行 NaN，不放行 ``±Inf``。NaN 在射电数据里有确定含义
    （被掩掉的频率通道 / 缺测样本），可以由预处理的 ``nan_policy`` 决定如何处理；
    ``Inf`` 没有对应的物理含义，且会让分位数和 min-max 归一化整体失效，
    因此在任何策略下都直接拒绝。
    """
    try:
        raw = np.asarray(data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if np.iscomplexobj(raw):
        # 科学数据若仍是复电压，静默丢弃虚部会改变物理含义，必须让调用者显式处理。
        raise ValueError(f"{name} must be real-valued; complex values are not supported")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if ndim is not None:
        valid_ndims = (ndim,) if isinstance(ndim, int) else ndim
        if array.ndim not in valid_ndims:
            expected = " or ".join(f"{value}D" for value in valid_ndims)
            raise ValueError(f"{name} must be {expected}")
    if allow_nan:
        if np.isinf(array).any():
            raise ValueError(f"{name} must not contain infinite values")
    elif not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_finite_array(
    data: Any,
    *,
    name: str = "data",
    ndim: int | tuple[int, ...] | None = None,
) -> np.ndarray:
    """把输入转换为非空、有限、实数 ``float64`` 数组。"""
    return _as_real_array(data, name=name, ndim=ndim, allow_nan=False)


def _immutable_array(data: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Return a C-contiguous array backed by an immutable bytes object.

    ``array.setflags(write=False)`` is reversible when an array owns a mutable
    allocation. Building the public snapshot from ``bytes`` makes NumPy's
    write-protection structural, so ``setflags(write=True)`` cannot reopen the
    provenance buffer for mutation.
    """
    array = np.asarray(data, dtype=dtype)
    if array.dtype.hasobject:
        raise TypeError("immutable snapshots do not support object arrays")
    # ``tobytes(order="C")`` serializes any strided layout in C order on its
    # own. A separate contiguity pass would add a second full-size copy for
    # exactly the transposed views ``_standard_layout`` builds from 3-D input,
    # which is where peak memory already matters most.
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype)
    return frozen.reshape(array.shape)


def _merge_settings(
    defaults: Mapping[str, Any],
    supplied: Mapping[str, Any] | None,
    *,
    field_name: str,
    unknown_label: str,
) -> dict[str, Any]:
    """Merge one public settings mapping over its defaults.

    Method parameters, preprocessing parameters, postprocessor parameters and
    the grouped perceptual voice/event mappings are all keyword containers with
    the same contract, so they share one container check, one string-key check,
    and one unknown-key report.
    """
    if supplied is None:
        return dict(defaults)
    if not isinstance(supplied, Mapping):
        raise ValueError(f"{field_name} must be a mapping or None")
    if any(not isinstance(key, str) for key in supplied):
        raise ValueError(f"{field_name} keys must be strings")
    unknown = sorted(set(supplied) - set(defaults))
    if unknown:
        joined = ", ".join(unknown)
        allowed = ", ".join(defaults)
        raise ValueError(f"{unknown_label}: {joined}; allowed: {allowed}")
    return {**defaults, **supplied}


def _positive_int(value: Any, *, name: str) -> int:
    """校验正整数；布尔值不能冒充 0/1。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    """校验非负整数；用于随机种子等允许为零的参数。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _finite_float(value: Any, *, name: str) -> float:
    """校验有限实数，并返回普通 ``float``。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive_float(value: Any, *, name: str) -> float:
    """校验大于零的有限实数。"""
    result = _finite_float(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be a finite number greater than 0")
    return result


def _boolean(value: Any, *, name: str) -> bool:
    """只接受 Python/NumPy 布尔值，避免字符串 ``"false"`` 被当作真。"""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _choice(value: Any, *, name: str, choices: set[str]) -> str:
    """Normalize and validate a public string choice."""
    joined = ", ".join(sorted(choices))
    if not isinstance(value, str):
        raise ValueError(f"{name} must be one of: {joined}")
    result = value.strip().lower().replace("-", "_").replace(" ", "_")
    if result not in choices:
        raise ValueError(f"{name} must be one of: {joined}")
    return result


def _peak_normalize(audio: np.ndarray, *, peak: float = 0.95) -> np.ndarray:
    """把有限的单声道或 samples x channels 音频归一化到 ``peak``。"""
    if not np.isfinite(peak) or not 0 < peak <= 1:
        raise ValueError("peak must be in the interval (0, 1]")
    result = _as_finite_array(audio, name="audio", ndim=(1, 2))
    current_peak = float(np.max(np.abs(result)))
    if current_peak > 0:
        result = result * (peak / current_peak)
    return result.astype(np.float32)


def require(module: str, extra: str) -> Any:
    """导入可选依赖；失败时给出对应 extra 的安装提示。

    Args:
        module: Module name to import.
        extra: pip install extra name (e.g., 'hifigan', 'musicnet').

    Returns:
        The imported module.

    Raises:
        ImportError: If the module is not installed.
    """
    try:
        # fromlist 确保 ``skimage.transform`` 这类点号路径返回目标子模块本身。
        return __import__(module, fromlist=["*"])
    except ImportError as err:
        raise ImportError(
            f"This feature requires '{module}'. Install with: "
            f"pip install radiosonify[{extra}]. Original error: {err}"
        ) from err
    except OSError as err:
        raise ImportError(
            f"Optional dependency '{module}' is installed but failed to load its binary "
            "libraries. Repair or reinstall that package in the active environment. "
            f"Original error: {err}"
        ) from err


# ---------- 科学数组变换 ----------


def _normalize_validated(data: np.ndarray) -> np.ndarray:
    """归一化已由公开入口验证的 ``float64`` 有限数组。"""
    dmin, dmax = float(data.min()), float(data.max())
    if dmax == dmin:
        return np.zeros_like(data)

    # 先缩放再求极差，避免 ``1e308 - (-1e308)`` 之类的有限输入发生溢出。
    scale = max(abs(dmin), abs(dmax))
    scaled = data / scale
    scaled_min = dmin / scale
    scaled_max = dmax / scale
    return (scaled - scaled_min) / (scaled_max - scaled_min)


def normalize(data: np.ndarray) -> np.ndarray:
    """把数组线性归一化到 ``[0, 1]``，常量数组返回零。"""
    return _normalize_validated(_as_finite_array(data))


def del_burst(data: np.ndarray, exposure_cut: int = 25) -> np.ndarray:
    """Clean burst data by clipping outliers and normalizing.

    Scales each column safely, divides by non-zero column means, clips to a
    percentile range, then normalizes to [0, 1].

    Args:
        data: 2D array (time x freq).
        exposure_cut: Percentile cut parameter.
    """
    warnings.warn(
        "del_burst() is deprecated and will be removed in RadioSonify 0.3; "
        "use preprocess() with explicit baseline, scale, and clipping settings",
        DeprecationWarning,
        stacklevel=2,
    )
    data = _as_finite_array(data, name="data", ndim=2)
    exposure_cut = _positive_int(exposure_cut, name="exposure_cut")
    if exposure_cut <= 1:
        raise ValueError("exposure_cut must be greater than 1")

    column_scale = np.max(np.abs(data), axis=0)
    normalized_columns = np.divide(
        data,
        column_scale,
        out=np.zeros_like(data),
        where=column_scale > 0,
    )
    normalized_mean = np.mean(normalized_columns, axis=0)
    safe_mean = np.abs(normalized_mean) > np.finfo(np.float64).eps

    # 均值接近零通常来自正负抵消；此时只保留按列峰值缩放后的形状，不再除以均值。
    scaled = normalized_columns.copy()
    np.divide(normalized_columns, normalized_mean, out=scaled, where=safe_mean)
    lower_percentile = 100.0 / exposure_cut
    upper_percentile = 100.0 * (exposure_cut - 1) / exposure_cut
    vmin = np.percentile(scaled, lower_percentile)
    vmax = np.percentile(scaled, upper_percentile)
    return _normalize_validated(np.clip(scaled, vmin, vmax))


def _rebin_axis(
    data: np.ndarray,
    target_bins: int,
    *,
    axis: int,
    nan_aware: bool = False,
) -> np.ndarray:
    """沿一个轴做等宽面积平均，不丢弃首尾样本。

    ``nan_aware=True`` treats NaNs as missing samples and renormalizes each
    target bin by the valid overlap. A target bin with no valid contribution
    remains NaN for the preprocessing mask policy to handle later.
    """
    source_bins = data.shape[axis]
    if target_bins == source_bins:
        return data

    moved = np.moveaxis(data, axis, 0)
    rebinned = np.empty((target_bins, *moved.shape[1:]), dtype=np.float64)

    # 把每个输入 bin 视为宽度为 1 的分段常量，在每个等宽目标区间内直接
    # 做加权面积平均。逐目标 bin 计算只需要一个小切片；旧的全轴累计和会
    # 对大型动态谱同时分配数个与输入等大的数组。
    edges = np.linspace(0.0, float(source_bins), target_bins + 1)
    # 两个切片的长度由 ``edges`` 的构造保证相等，因此不需要 ``strict=``；
    # 该关键字是 Python 3.10 才引入的，而本包声明支持 3.9。
    for target_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        first = int(np.floor(left))
        stop = int(np.ceil(right))
        source_indices = np.arange(first, stop, dtype=np.float64)
        weights = np.minimum(source_indices + 1.0, right) - np.maximum(
            source_indices,
            left,
        )
        chunk = moved[first:stop]
        if nan_aware:
            valid = ~np.isnan(chunk)
            weight_shape = (len(weights),) + (1,) * (chunk.ndim - 1)
            expanded_weights = weights.reshape(weight_shape)
            numerator = np.sum(
                np.where(valid, chunk, 0.0) * expanded_weights,
                axis=0,
            )
            denominator = np.sum(valid * expanded_weights, axis=0)
            rebinned[target_index] = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan),
                where=denominator > 0,
            )
        else:
            rebinned[target_index] = np.tensordot(
                weights,
                chunk,
                axes=(0, 0),
            ) / (right - left)
    return np.moveaxis(rebinned, 0, axis)


def rebin_spectrogram(
    data: np.ndarray,
    time_bins: int | None = None,
    freq_bins: int | None = None,
) -> np.ndarray:
    """按等宽面积平均把二维动态谱重分箱。

    Note:
        目标尺寸不整除原尺寸时，会按边界重叠比例分配样本。整个时间和频率范围都会
        被使用，不再截断尾部数据。

    Args:
        data: 2D array (time x freq).
        time_bins: Target number of time bins. None keeps original.
        freq_bins: Target number of freq bins. None keeps original.
    """
    warnings.warn(
        "rebin_spectrogram() is deprecated and will be removed in RadioSonify 0.3; "
        "use preprocess(time_rebin=..., feature_rebin=...), which resizes in both "
        "directions and records the effective settings in the result",
        DeprecationWarning,
        stacklevel=2,
    )
    result = _as_finite_array(data, name="data", ndim=2)
    t0, f0 = result.shape

    if time_bins is not None:
        time_bins = _positive_int(time_bins, name="time_bins")
        if time_bins > t0:
            raise ValueError(f"time_bins ({time_bins}) cannot exceed input time dimension ({t0})")

    if freq_bins is not None:
        freq_bins = _positive_int(freq_bins, name="freq_bins")
        if freq_bins > f0:
            raise ValueError(
                f"freq_bins ({freq_bins}) cannot exceed input frequency dimension ({f0})"
            )

    if time_bins is not None:
        result = _rebin_axis(result, time_bins, axis=0)
    if freq_bins is not None:
        result = _rebin_axis(result, freq_bins, axis=1)

    return result


def to_profile(data: np.ndarray) -> np.ndarray:
    """把一维数组或二维矩阵转换为一维时间轮廓。

    二维输入沿特征轴求均值。时间轴的重分箱属于统一预处理的 ``time_rebin``，
    不再在这里做。
    """
    data = _as_finite_array(data, name="data", ndim=(1, 2))
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    return data


def _interpolate_cyclic_profile(profile: np.ndarray, *, n_samples: int) -> np.ndarray:
    """把轮廓循环插值到指定采样数。

    轮廓是分箱积分数据：``N`` 个 bin 覆盖 ``N`` 个等宽区间，因此最后一个 bin
    之后接回第一个 bin。这让预处理阶段做的 ``repeat`` 次拼接在每个接缝处都保持
    连续 —— 实测硬拼接会把真实折叠轮廓 4 kHz 以上的能量抬高约 6.5 倍，
    即可听的咔哒声。
    """
    profile = _as_finite_array(profile, name="profile", ndim=1)
    n_samples = _positive_int(n_samples, name="n_samples")

    total_points = len(profile)
    if total_points > 2**53:
        raise ValueError("profile is too long for exact float64 interpolation")

    positions = np.arange(n_samples, dtype=np.float64) * (total_points / n_samples)
    left = np.floor(positions).astype(np.int64)
    fraction = positions - left
    return (
        profile[left % total_points] * (1.0 - fraction)
        + profile[(left + 1) % total_points] * fraction
    )


# ---------- WAV 输出 ----------


def _wav_output_path(path: str | Path) -> Path:
    """校验 WAV 输出路径，但不创建目录或写文件。"""
    try:
        output_path = Path(path)
    except (TypeError, ValueError) as exc:
        raise ValueError("path must name a WAV file") from exc
    if not output_path.name or output_path.suffix.lower() != ".wav":
        raise ValueError("path must name a file with the .wav extension")
    if output_path.exists() and not output_path.is_file():
        raise ValueError(f"path points to a directory, not a WAV file: {output_path}")
    if output_path.is_symlink() and not output_path.exists():
        raise ValueError(f"path points to a broken symbolic link: {output_path}")

    # Fail before synthesis when an existing parent component is a file. The
    # remaining missing directories are safe for save_audio() to create later.
    existing_parent = output_path.parent
    while not existing_parent.exists() and existing_parent.parent != existing_parent:
        existing_parent = existing_parent.parent
    if existing_parent.exists() and not existing_parent.is_dir():
        raise ValueError(f"WAV output parent is not a directory: {existing_parent}")
    return output_path


def save_audio(audio: np.ndarray, sr: int, path: str | Path) -> None:
    """把有限、未削波的单/多声道音频保存为 PCM16 WAV。"""
    sr = _positive_int(sr, name="sr")
    data = _as_finite_array(audio, name="audio", ndim=(1, 2))
    if float(np.max(np.abs(data))) > 1.0 + 1e-7:
        raise ValueError("audio must stay within [-1, 1] to avoid PCM clipping")

    output_path = _wav_output_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, data, sr, format="WAV", subtype="PCM_16")
