"""三维分层矩阵的感知滤波组合成与立体声空间化。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._perceptual import (
    _condition_synthesized_audio,
    _settings_from_mapping,
    _synthesize_prepared,
)
from ._perceptual_config import PERCEPTUAL_DEFAULT_DURATION, PERCEPTUAL_DEFAULTS
from .core import (
    _finite_float,
    _wav_output_path,
    save_audio,
)
from .preprocessing import _as_normalized_array
from .timing import duration_to_samples


def _positions_or_gains(
    values: Sequence[float] | np.ndarray | None,
    *,
    count: int,
    name: str,
    lower: float,
    upper: float | None,
) -> np.ndarray | None:
    if values is None:
        return None
    message = f"{name} must be a reusable sequence of {count} finite numbers"
    if isinstance(values, (str, bytes, bytearray, memoryview)) or not isinstance(
        values, (Sequence, np.ndarray)
    ):
        raise ValueError(message)
    try:
        raw = list(values)
    except TypeError as exc:
        raise ValueError(message) from exc
    if len(raw) != count:
        raise ValueError(f"{name} must contain exactly one value per layer ({count})")
    result = np.asarray([_finite_float(value, name=name) for value in raw])
    if np.any(result < lower) or (upper is not None and np.any(result > upper)):
        interval = f"[{lower:g}, {upper:g}]" if upper is not None else f"[{lower:g}, infinity)"
        raise ValueError(f"{name} values must lie in {interval}")
    return result


def spatial_sonify(
    data: np.ndarray,
    sr: int = PERCEPTUAL_DEFAULTS["sr"],
    duration: float = PERCEPTUAL_DEFAULT_DURATION,
    min_freq: float = PERCEPTUAL_DEFAULTS["min_freq"],
    max_freq: float = PERCEPTUAL_DEFAULTS["max_freq"],
    n_bands: int | None = PERCEPTUAL_DEFAULTS["n_bands"],
    value_scale: str = PERCEPTUAL_DEFAULTS["value_scale"],
    gamma: float = PERCEPTUAL_DEFAULTS["gamma"],
    frequency_order: str = PERCEPTUAL_DEFAULTS["frequency_order"],
    frequency_scale: str = PERCEPTUAL_DEFAULTS["frequency_scale"],
    timbre: str = PERCEPTUAL_DEFAULTS["timbre"],
    mapping_level_db: float = PERCEPTUAL_DEFAULTS["mapping_level_db"],
    ambient_level_db: float = PERCEPTUAL_DEFAULTS["ambient_level_db"],
    voice_params: Mapping[str, Any] | None = PERCEPTUAL_DEFAULTS["voice_params"],
    event_voice: str = PERCEPTUAL_DEFAULTS["event_voice"],
    event_params: Mapping[str, Any] | None = PERCEPTUAL_DEFAULTS["event_params"],
    attack_ms: float = PERCEPTUAL_DEFAULTS["attack_ms"],
    release_ms: float = PERCEPTUAL_DEFAULTS["release_ms"],
    loudness_compensation_db: float = PERCEPTUAL_DEFAULTS["loudness_compensation_db"],
    rms_limit_dbfs: float = PERCEPTUAL_DEFAULTS["rms_limit_dbfs"],
    peak_limit_dbfs: float = PERCEPTUAL_DEFAULTS["peak_limit_dbfs"],
    pan_positions: Sequence[float] | np.ndarray | None = None,
    layer_gains: Sequence[float] | np.ndarray | None = None,
    output: str | Path | None = None,
) -> tuple[np.ndarray, int]:
    """把预处理到 ``[0,1]`` 的三维张量逐层合成并按位置混成立体声。

    输入必须已经是标准布局 ``(layer, time, feature)``；轴顺序由
    ``SonificationInput`` 的 ``layer_axis`` / ``time_axis`` 负责，不是方法参数。
    层可以是 I/Q/U/V、图像通道、不同传感器或任意其他并列二维量；实现不识别也
    不硬编码层的科学含义。``pan_positions`` 使用 ``-1``（左）到 ``+1``（右）的
    恒功率声像。

    统一 API 默认对三维输入按层归一化，因此各层的可听响度不再被最强的一层
    压掉；层与层之间真实的科学强度差应当通过 ``layer_gains`` 显式表达。
    """
    arguments = locals()
    output_path = None if output is None else _wav_output_path(output)
    layers = _as_normalized_array(data, name="data", ndim=3)
    settings = _settings_from_mapping(arguments)

    layer_count = layers.shape[0]
    pans = _positions_or_gains(
        pan_positions,
        count=layer_count,
        name="pan_positions",
        lower=-1.0,
        upper=1.0,
    )
    if pans is None:
        pans = np.zeros(1) if layer_count == 1 else np.linspace(-1.0, 1.0, layer_count)
    gains = _positions_or_gains(
        layer_gains,
        count=layer_count,
        name="layer_gains",
        lower=0.0,
        upper=None,
    )
    if gains is None:
        gains = np.ones(layer_count)

    stereo = np.zeros(
        (duration_to_samples(settings.duration, settings.sr), 2),
        dtype=np.float64,
    )
    event_rate_scale = 1.0 / layer_count
    for layer, (pan, gain) in enumerate(zip(pans, gains)):
        mono = _synthesize_prepared(
            layers[layer],
            settings=settings,
            event_rate_scale=event_rate_scale,
        )
        angle = (pan + 1.0) * np.pi / 4.0
        contribution = gain * mono[:, None] * np.array([np.cos(angle), np.sin(angle)])
        stereo += contribution
    audio = _condition_synthesized_audio(
        stereo,
        sr=settings.sr,
        rms_limit_dbfs=settings.rms_limit_dbfs,
        peak_limit_dbfs=settings.peak_limit_dbfs,
    )
    if output_path is not None:
        save_audio(audio, settings.sr, output_path)
    return audio, settings.sr


__all__ = ["spatial_sonify"]
