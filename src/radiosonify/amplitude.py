"""用一维轮廓调制固定基频及其谐波的振幅。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import (
    _finite_float,
    _interpolate_cyclic_profile,
    _peak_normalize,
    _positive_float,
    _positive_int,
    _wav_output_path,
    save_audio,
    to_profile,
)
from .preprocessing import _as_normalized_array
from .timing import duration_to_samples

_DEFAULT_COMPRESSION = 0.0


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
    output: str | Path | None = None,
    *,
    compression: float = _DEFAULT_COMPRESSION,
    harmonics: int = 4,
    harmonic_decay: float = 1.0,
) -> tuple[np.ndarray, int]:
    """把轮廓强度映射为固定基频及其谐波的响度。

    Repetition and rebinning belong to :func:`radiosonify.preprocess`; the
    number of pulse cycles heard in ``duration`` follows from how many cycles
    the preprocessed profile already contains.

    Args:
        data: Preprocessed ``[0, 1]`` 1D profile, or a 2D matrix whose feature
            axis has already been reduced.
        sr: Sample rate in Hz.
        duration: Output audio duration in seconds.
        freq: Fundamental carrier frequency in Hz.
        compression: Log compression strength. 0 is linear; larger values make
            weak profile structure louder relative to the peak.
        output: Path to save WAV file. None = don't save.
        harmonics: Maximum number of integer harmonics, including the fundamental.
            Partials at or above Nyquist are omitted automatically.
        harmonic_decay: Harmonic ``h`` receives weight ``1 / h**harmonic_decay``.
            Set ``harmonics=1`` for the historical single-sine carrier.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    output_path = None if output is None else _wav_output_path(output)
    sr = _positive_int(sr, name="sr")
    duration = _positive_float(duration, name="duration")
    freq = _positive_float(freq, name="freq")
    compression = _validate_compression(compression)
    harmonics = _positive_int(harmonics, name="harmonics")
    harmonic_decay = _finite_float(harmonic_decay, name="harmonic_decay")
    if harmonic_decay < 0:
        raise ValueError("harmonic_decay must be a finite number greater than or equal to 0")
    if freq >= sr / 2:
        raise ValueError(f"freq must be between 0 and the Nyquist frequency ({sr / 2:g} Hz)")

    n_samples = duration_to_samples(duration, sr)
    profile = to_profile(data)
    profile = _as_normalized_array(profile, name="profile", ndim=1)

    if len(profile) < 2 or float(np.ptp(profile)) == 0:
        audio = np.zeros(n_samples, dtype=np.float32)
    else:
        # 先压缩强度动态范围，使弱结构仍能在振幅包络中被听见。
        profile = _compress_profile(profile, compression)

        t = np.linspace(0.0, duration, n_samples, endpoint=False)
        envelope = _interpolate_cyclic_profile(profile, n_samples=n_samples)

        # 所有谐波共享同一个包络；因此音色更饱满，但轮廓不会被重新编码为音高。
        # 先按 Nyquist 算出真正可听的上限，避免一个极大的 ``harmonics`` 请求
        # 为随后必然丢弃的超声分音分配同样巨大的临时数组。
        nyquist = sr / 2
        try:
            highest_requested_frequency = freq * harmonics
        except OverflowError:
            highest_requested_frequency = np.inf
        if np.isfinite(highest_requested_frequency) and highest_requested_frequency < nyquist:
            audible_harmonics = harmonics
        else:
            nyquist_ratio = nyquist / freq
            if not np.isfinite(nyquist_ratio) or nyquist_ratio > np.iinfo(np.intp).max:
                raise ValueError("freq and harmonics request too many audible harmonics")
            audible_harmonics = min(harmonics, int(np.ceil(nyquist_ratio) - 1))
        harmonic_numbers = np.arange(1, audible_harmonics + 1, dtype=np.float64)
        weights = 1.0 / harmonic_numbers**harmonic_decay
        carrier = np.zeros(n_samples, dtype=np.float64)
        for harmonic, weight in zip(harmonic_numbers, weights):
            carrier += weight * np.sin(2.0 * np.pi * freq * harmonic * t)
        carrier /= np.sum(weights)
        audio = _peak_normalize(envelope * carrier, peak=0.9)

    if output_path is not None:
        save_audio(audio, sr, output_path)

    return audio, sr
