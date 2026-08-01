"""方法 2：用时间轮廓调制正弦载波的振幅。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import (
    _finite_float,
    _interpolate_repeated_profile,
    _peak_normalize,
    _positive_float,
    _positive_int,
    _wav_output_path,
    save_audio,
    to_profile,
)
from .timing import duration_to_samples

_DEFAULT_COMPRESSION = 99.0


def _validate_compression(compression: float) -> float:
    compression = _finite_float(compression, name="compression")
    if compression < 0:
        raise ValueError("compression must be a finite number greater than or equal to 0")
    return compression


def _compress_profile(profile: np.ndarray, compression: float) -> np.ndarray:
    """把归一化强度映射为可控的对数包络；0 表示线性。"""
    if compression == 0:
        return profile
    return np.log1p(compression * profile) / np.log1p(compression)


def amplitude_modulate(
    data: np.ndarray,
    sr: int = 48000,
    duration: float = 2.0,
    freq: float = 1000.0,
    repeat: int = 1,
    time_downsample: int | None = None,
    output: str | Path | None = None,
    *,
    compression: float = _DEFAULT_COMPRESSION,
) -> tuple[np.ndarray, int]:
    """把脉冲轮廓的强度映射为正弦载波响度。

    Args:
        data: 1D profile or 2D spectrogram (time x freq).
        sr: Sample rate in Hz.
        duration: Output audio duration in seconds.
        freq: Carrier sine wave frequency in Hz.
        repeat: Number of profile cycles represented in ``duration``.
        compression: Log compression strength. 0 is linear; larger values make
            weak profile structure louder relative to the peak.
        time_downsample: Downsample factor. None = no downsampling.
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    output_path = None if output is None else _wav_output_path(output)
    sr = _positive_int(sr, name="sr")
    duration = _positive_float(duration, name="duration")
    freq = _positive_float(freq, name="freq")
    repeat = _positive_int(repeat, name="repeat")
    compression = _validate_compression(compression)
    if freq >= sr / 2:
        raise ValueError(f"freq must be between 0 and the Nyquist frequency ({sr / 2:g} Hz)")

    n_samples = duration_to_samples(duration, sr)
    profile = to_profile(data, downsample=time_downsample)

    if len(profile) < 2 or float(np.ptp(profile)) == 0:
        audio = np.zeros(n_samples, dtype=np.float32)
    else:
        # 先压缩强度动态范围，使弱结构仍能在振幅包络中被听见。
        profile = (profile - profile.min()) / (profile.max() - profile.min())
        profile = _compress_profile(profile, compression)

        t = np.linspace(0.0, duration, n_samples, endpoint=False)
        envelope = _interpolate_repeated_profile(
            profile,
            repeat=repeat,
            n_samples=n_samples,
        )

        # 轮廓只控制响度；载波频率保持为用户指定的可听频率。
        carrier = np.sin(2.0 * np.pi * freq * t)
        audio = _peak_normalize(envelope * carrier, peak=0.9)

    if output_path is not None:
        save_audio(audio, sr, output_path)

    return audio, sr
