"""方法 1：把时间轮廓直接插值为波形，可选乐器脉冲响应卷积。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .core import (
    _interpolate_cyclic_profile,
    _peak_normalize,
    _positive_float,
    _positive_int,
    _wav_output_path,
    save_audio,
    to_profile,
)
from .hub import get_instrument_path
from .preprocessing import _as_normalized_array
from .timing import _resample_audio_rate, duration_to_samples

_WAVE_PEAK = 0.95


def _read_wave(file: str) -> tuple[np.ndarray, int]:
    """读取乐器 WAV，混合为单声道并返回原采样率。"""
    wave_data, source_sr = sf.read(file, always_2d=True)
    if wave_data.size == 0:
        raise ValueError(f"instrument WAV is empty: {file}")
    if not np.all(np.isfinite(wave_data)):
        raise ValueError(f"instrument WAV contains non-finite samples: {file}")
    mono = np.mean(wave_data.astype(np.float64), axis=1)
    return mono, _positive_int(source_sr, name="instrument sample rate")


def profile_to_wave(
    data: np.ndarray,
    sr: int = 48000,
    duration: float = 10.0,
    instrument: str | None = "violin",
    output: str | Path | None = None,
) -> tuple[np.ndarray, int]:
    """把脉冲轮廓转换为可听波形。

    The profile is interpolated to the target duration and optionally convolved
    with a deterministic synthesized instrument response. Repetition, rebinning
    and any other change to the data belong to :func:`radiosonify.preprocess`.

    Args:
        data: Preprocessed ``[0, 1]`` 1D profile, or a 2D matrix whose feature
            axis has already been reduced.
        sr: Sample rate in Hz.
        duration: Output audio duration in seconds.
        instrument: Instrument for convolution ('violin', 'piano', or None).
            This entry point keeps 'violin' for compatibility. The registered
            unified-API default is None, so :func:`radiosonify.sonify` treats
            the instrument response as opt-in; ``radiosonify list-settings``
            prints the registered value.
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    output_path = None if output is None else _wav_output_path(output)
    sr = _positive_int(sr, name="sr")
    duration = _positive_float(duration, name="duration")
    if instrument not in (None, "violin", "piano"):
        raise ValueError("instrument must be 'violin', 'piano', or None")

    n_samples = duration_to_samples(duration, sr)

    profile = to_profile(data)
    profile = _as_normalized_array(profile, name="profile", ndim=1)
    if len(profile) < 2 or float(np.ptp(profile)) == 0:
        audio = np.zeros(n_samples, dtype=np.float32)
    else:
        # 先映射到双极性范围，确保轮廓的相对形状成为声压波形而不是纯 DC。
        wave_raw = _interpolate_cyclic_profile(profile, n_samples=n_samples)
        wave_raw = wave_raw * 2.0 - 1.0

        if instrument is not None:
            instrument_path = get_instrument_path(instrument)
            sound, source_sr = _read_wave(instrument_path)
            sound = _resample_audio_rate(sound, source_sr, sr)
            sound = sound - np.mean(sound)
            scale = float(np.sum(np.abs(sound)))
            if scale <= 1e-12:
                raise ValueError(f"instrument WAV has no usable AC signal: {instrument_path}")
            # L1 归一化脉冲响应，防止卷积长度改变整体增益。这里必须使用因果截取；
            # ``mode="same"`` 会从完整卷积的中心取样，几毫秒的轮廓可能只截到
            # 乐器样本中近似常量的一小段，最终去直流后就会变成静音。
            convolved = signal.fftconvolve(wave_raw, sound / scale, mode="full")
            wave_raw = convolved[: len(wave_raw)]

        audio = _peak_normalize(wave_raw, peak=_WAVE_PEAK)

    if output_path is not None:
        save_audio(audio, sr, output_path)

    return audio, sr
