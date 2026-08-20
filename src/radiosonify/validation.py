"""Shared validation and immutable-value helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np


def _as_real_array(
    data: Any,
    *,
    name: str = "data",
    ndim: int | tuple[int, ...] | None = None,
    allow_nan: bool = False,
) -> np.ndarray:
    """Convert input to a non-empty real ``float64`` array.

    ``allow_nan=True`` permits missing values while still rejecting infinities.
    Complex input is rejected so scientific phase information is never silently
    discarded.
    """
    try:
        raw = np.asarray(data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if np.iscomplexobj(raw):
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
    """Convert input to a non-empty, finite, real ``float64`` array."""
    return _as_real_array(data, name=name, ndim=ndim, allow_nan=False)


def _immutable_array(data: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Return a C-layout array backed by an immutable bytes object."""
    array = np.asarray(data, dtype=dtype)
    if array.dtype.hasobject:
        raise TypeError("immutable snapshots do not support object arrays")
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype)
    return frozen.reshape(array.shape)


def _freeze_value(value: Any) -> Any:
    """Recursively snapshot one value into immutable containers and buffers."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return _freeze_value(value.tolist())
        return _immutable_array(value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, Sequence):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a recursively immutable snapshot of a string-keyed mapping."""
    return MappingProxyType({key: _freeze_value(value) for key, value in values.items()})


def _merge_settings(
    defaults: Mapping[str, Any],
    supplied: Mapping[str, Any] | None,
    *,
    field_name: str,
    unknown_label: str,
) -> dict[str, Any]:
    """Merge a validated public settings mapping over immutable defaults."""
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
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be a finite number greater than 0")
    return result


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _choice(value: Any, *, name: str, choices: set[str]) -> str:
    joined = ", ".join(sorted(choices))
    if not isinstance(value, str):
        raise ValueError(f"{name} must be one of: {joined}")
    result = value.strip().lower().replace("-", "_").replace(" ", "_")
    if result not in choices:
        raise ValueError(f"{name} must be one of: {joined}")
    return result
