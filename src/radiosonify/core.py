"""RadioSonify 的公共数值校验、数据变换与 WAV I/O 工具。"""

from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# ---------- 数值输入与参数校验 ----------


def _as_finite_array(
    data: Any,
    *,
    name: str = "data",
    ndim: int | tuple[int, ...] | None = None,
) -> np.ndarray:
    """把输入转换为非空、有限、实数 ``float64`` 数组。"""
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
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_int(value: int, *, name: str) -> int:
    """校验正整数；布尔值不能冒充 0/1。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_int(value: int, *, name: str) -> int:
    """校验非负整数；用于随机种子等允许为零的参数。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _finite_float(value: float, *, name: str) -> float:
    """校验有限实数，并返回普通 ``float``。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive_float(value: float, *, name: str) -> float:
    """校验大于零的有限实数。"""
    result = _finite_float(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be a finite number greater than 0")
    return result


def _boolean(value: bool, *, name: str) -> bool:
    """只接受 Python/NumPy 布尔值，避免字符串 ``"false"`` 被当作真。"""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _peak_normalize(audio: np.ndarray, *, peak: float = 0.95) -> np.ndarray:
    """把单声道有限音频的绝对峰值归一化到 ``peak``。"""
    if not np.isfinite(peak) or not 0 < peak <= 1:
        raise ValueError("peak must be in the interval (0, 1]")
    result = _as_finite_array(audio, name="audio", ndim=1)
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


def _validate_exposure_cut(exposure_cut: int) -> int:
    exposure_cut = _positive_int(exposure_cut, name="exposure_cut")
    if exposure_cut <= 1:
        raise ValueError("exposure_cut must be greater than 1")
    return exposure_cut


def _del_burst_validated(data: np.ndarray, exposure_cut: int) -> np.ndarray:
    """清理已验证的二维数组，避免组合流水线再次全量扫描。"""
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


def del_burst(data: np.ndarray, exposure_cut: int = 25) -> np.ndarray:
    """Clean burst data by clipping outliers and normalizing.

    Scales each column safely, divides by non-zero column means, clips to a
    percentile range, then normalizes to [0, 1].

    Args:
        data: 2D array (time x freq).
        exposure_cut: Percentile cut parameter.
    """
    data = _as_finite_array(data, name="data", ndim=2)
    return _del_burst_validated(data, _validate_exposure_cut(exposure_cut))


def _rebin_axis(data: np.ndarray, target_bins: int, *, axis: int) -> np.ndarray:
    """沿一个轴做等宽面积平均，不丢弃首尾样本。"""
    source_bins = data.shape[axis]
    if target_bins == source_bins:
        return data

    moved = np.moveaxis(data, axis, 0)
    cumulative = np.concatenate(
        (np.zeros((1, *moved.shape[1:]), dtype=np.float64), np.cumsum(moved, axis=0)),
        axis=0,
    )

    # 把每个输入 bin 视为宽度为 1 的分段常量；在等宽目标边界上计算积分差。
    edges = np.linspace(0.0, float(source_bins), target_bins + 1)
    whole = np.floor(edges).astype(np.intp)
    fractions = (edges - whole).reshape((-1,) + (1,) * (moved.ndim - 1))
    partial_rows = moved[np.minimum(whole, source_bins - 1)]
    integrals = cumulative[whole] + partial_rows * fractions
    rebinned = np.diff(integrals, axis=0) / (source_bins / target_bins)
    return np.moveaxis(rebinned, 0, axis)


def _rebin_spectrogram_validated(
    result: np.ndarray,
    time_bins: int | None,
    freq_bins: int | None,
) -> np.ndarray:
    """重分箱已验证的二维数组，同时仍严格校验目标尺寸。"""
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
    result = _as_finite_array(data, name="data", ndim=2)
    return _rebin_spectrogram_validated(result, time_bins, freq_bins)


def to_profile(
    data: np.ndarray,
    downsample: int | None = None,
) -> np.ndarray:
    """把一维数组或二维动态谱转换为一维时间轮廓。

    二维输入沿频率轴求均值。``downsample`` 是近似分组因子，输出长度仍为
    ``floor(length / downsample)``，但通过等宽面积平均使用全部输入样本。
    """
    data = _as_finite_array(data, name="data", ndim=(1, 2))
    if data.ndim == 2:
        data = np.mean(data, axis=1)

    if downsample is not None:
        downsample = _positive_int(downsample, name="downsample")
        if downsample > len(data):
            raise ValueError(
                f"downsample ({downsample}) cannot exceed profile length ({len(data)})"
            )
        if downsample > 1:
            target_bins = len(data) // downsample
            data = _rebin_axis(data, target_bins, axis=0)

    return data


def _interpolate_repeated_profile(
    profile: np.ndarray,
    *,
    repeat: int,
    n_samples: int,
) -> np.ndarray:
    """把轮廓插值到指定采样数，不创建巨型 ``tile`` 中间数组。

    轮廓是分箱积分数据：``N`` 个 bin 覆盖 ``N`` 个等宽区间，因此最后一个 bin
    之后接回第一个 bin。这同时让 ``repeat`` 次拼接在接缝处保持连续 —— 实测硬
    拼接会把真实折叠轮廓 4 kHz 以上的能量抬高约 6.5 倍，即可听的咔哒声。
    """
    profile = _as_finite_array(profile, name="profile", ndim=1)
    repeat = _positive_int(repeat, name="repeat")
    n_samples = _positive_int(n_samples, name="n_samples")

    total_points = len(profile) * repeat
    if total_points > 2**53:
        raise ValueError("repeat is too large for exact float64 profile interpolation")

    positions = np.arange(n_samples, dtype=np.float64) * (total_points / n_samples)
    left = np.floor(positions).astype(np.int64)
    fraction = positions - left
    return (
        profile[left % len(profile)] * (1.0 - fraction)
        + profile[(left + 1) % len(profile)] * fraction
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
