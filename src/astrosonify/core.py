"""Shared utilities for AstroSonify."""

from __future__ import annotations

import numpy as np
import soundfile as sf


def normalize(data: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    data = data.astype(np.float64)
    dmin, dmax = data.min(), data.max()
    if dmax == dmin:
        return np.zeros_like(data)
    return (data - dmin) / (dmax - dmin)


def del_burst(data: np.ndarray, exposure_cut: int = 25) -> np.ndarray:
    """Clean burst data by clipping outliers and normalizing.

    Divides by column mean, clips to percentile range, normalizes to [0, 1].

    Args:
        data: 2D array (time x freq).
        exposure_cut: Percentile cut parameter.
    """
    data = data.astype(np.float64)
    h, w = data.shape
    col_mean = np.mean(data, axis=0)
    col_mean[col_mean == 0] = 1.0
    data = data / col_mean
    flat = np.sort(data.flatten())
    vmin = flat[int(h * w / exposure_cut)]
    vmax = flat[int(h * w / exposure_cut * (exposure_cut - 1))]
    data = np.clip(data, vmin, vmax)
    return normalize(data)


def rebin_spectrogram(
    data: np.ndarray,
    time_bins: int | None = None,
    freq_bins: int | None = None,
) -> np.ndarray:
    """Rebin a 2D spectrogram by averaging adjacent bins.

    Args:
        data: 2D array (time x freq).
        time_bins: Target number of time bins. None keeps original.
        freq_bins: Target number of freq bins. None keeps original.
    """
    if data.ndim != 2:
        raise ValueError("rebin_spectrogram requires 2D input array")

    result = data.astype(np.float64)

    if time_bins is not None and time_bins != result.shape[0]:
        t = result.shape[0]
        usable = (t // time_bins) * time_bins
        result = result[:usable].reshape(time_bins, -1, result.shape[1]).mean(axis=1)

    if freq_bins is not None and freq_bins != result.shape[1]:
        f = result.shape[1]
        usable = (f // freq_bins) * freq_bins
        result = result[:, :usable].reshape(result.shape[0], freq_bins, -1).mean(axis=2)

    return result


def to_profile(
    data: np.ndarray,
    downsample: int | None = None,
) -> np.ndarray:
    """Convert data to 1D pulse profile.

    If 2D, averages along frequency axis (axis=1).
    """
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    elif data.ndim != 1:
        raise ValueError("to_profile expects 1D or 2D input")

    data = data.astype(np.float64)

    if downsample is not None and downsample > 1:
        usable = (len(data) // downsample) * downsample
        data = data[:usable].reshape(-1, downsample).mean(axis=1)

    return data


def save_audio(audio: np.ndarray, sr: int, path: str) -> None:
    """Save audio array to WAV file."""
    sf.write(path, audio, sr)
