"""Method-independent numerical array operations."""

from __future__ import annotations

import numpy as np

from .validation import _as_finite_array, _positive_int

_MAD_TO_GAUSSIAN_SIGMA = 1.4826


def _normalize_validated(data: np.ndarray) -> np.ndarray:
    dmin, dmax = float(data.min()), float(data.max())
    if dmax == dmin:
        return np.zeros_like(data)
    scale = max(abs(dmin), abs(dmax))
    scaled = data / scale
    return (scaled - dmin / scale) / (dmax / scale - dmin / scale)


def normalize(data: np.ndarray) -> np.ndarray:
    """Linearly normalize a finite numerical array to ``[0, 1]``."""
    return _normalize_validated(_as_finite_array(data))


def _rebin_axis(
    data: np.ndarray,
    target_bins: int,
    *,
    axis: int,
    nan_aware: bool = False,
) -> np.ndarray:
    """Area-average one axis into equally wide target bins."""
    source_bins = data.shape[axis]
    if target_bins == source_bins:
        return data

    moved = np.moveaxis(data, axis, 0)
    rebinned = np.empty((target_bins, *moved.shape[1:]), dtype=np.float64)
    edges = np.linspace(0.0, float(source_bins), target_bins + 1)
    for target_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        first = int(np.floor(left))
        stop = int(np.ceil(right))
        source_indices = np.arange(first, stop, dtype=np.float64)
        weights = np.minimum(source_indices + 1.0, right) - np.maximum(source_indices, left)
        chunk = moved[first:stop]
        if nan_aware:
            valid = ~np.isnan(chunk)
            weight_shape = (len(weights),) + (1,) * (chunk.ndim - 1)
            expanded_weights = weights.reshape(weight_shape)
            numerator = np.sum(np.where(valid, chunk, 0.0) * expanded_weights, axis=0)
            denominator = np.sum(valid * expanded_weights, axis=0)
            rebinned[target_index] = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan),
                where=denominator > 0,
            )
        else:
            rebinned[target_index] = np.tensordot(weights, chunk, axes=(0, 0)) / (right - left)
    return np.moveaxis(rebinned, 0, axis)


def to_profile(data: np.ndarray) -> np.ndarray:
    """Convert a profile or time-by-feature matrix to a time profile."""
    result = _as_finite_array(data, name="data", ndim=(1, 2))
    return np.mean(result, axis=1) if result.ndim == 2 else result


def _interpolate_cyclic_profile(profile: np.ndarray, *, n_samples: int) -> np.ndarray:
    """Interpolate binned profile values cyclically across a sample grid."""
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


__all__ = ["normalize", "to_profile"]
