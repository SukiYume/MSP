"""统一声化 API 的科学数据类型与不可变输入快照。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .validation import _as_real_array, _immutable_array, _positive_float


class DataType(str, Enum):
    """MSP 接受的一维轮廓、二维矩阵和三维分层矩阵。

    成员刻意只有三个，与输入维数一一对应。``dynamic_spectrum`` 等领域名称是
    :data:`_DATA_TYPE_ALIASES` 里的别名，而不是独立成员 —— 曾经把它写成同值的
    第二个成员，Python 会把它折叠成 ``MATRIX`` 的别名，于是注册表里
    ``input_types=(MATRIX,)`` 和 ``input_types=(DYNAMIC_SPECTRUM,)`` 看似不同、
    实则完全一致，是有误导性的假区分。
    """

    PROFILE = "profile"
    MATRIX = "matrix"
    LAYERED_MATRIX = "layered_matrix"

    def __str__(self) -> str:
        return self.value


_DATA_TYPE_ALIASES = {
    "profile": DataType.PROFILE,
    "pulse_profile": DataType.PROFILE,
    "dynamic_spectrum": DataType.MATRIX,
    "dynamic_spectra": DataType.MATRIX,
    "spectrogram": DataType.MATRIX,
    "matrix": DataType.MATRIX,
    "array_2d": DataType.MATRIX,
    "image": DataType.MATRIX,
    "layered_matrix": DataType.LAYERED_MATRIX,
    "matrix_stack": DataType.LAYERED_MATRIX,
    "image_stack": DataType.LAYERED_MATRIX,
    "cube": DataType.LAYERED_MATRIX,
    "iquv": DataType.LAYERED_MATRIX,
}


def parse_data_type(value: DataType | str) -> DataType:
    """把公开字符串及别名解析为标准 :class:`DataType`。"""
    if isinstance(value, DataType):
        return value
    if not isinstance(value, str):
        raise ValueError("data_type must be 'profile', 'matrix', or 'layered_matrix'")
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _DATA_TYPE_ALIASES[key]
    except KeyError as exc:
        raise ValueError("data_type must be 'profile', 'matrix', or 'layered_matrix'") from exc


def infer_data_type(data: np.ndarray) -> DataType:
    """按数组维数推断轮廓、矩阵或分层矩阵。"""
    array = np.asarray(data)
    if array.ndim == 1:
        return DataType.PROFILE
    if array.ndim == 2:
        return DataType.MATRIX
    if array.ndim == 3:
        return DataType.LAYERED_MATRIX
    raise ValueError(
        f"cannot infer data_type from a {array.ndim}D array; "
        "MSP accepts 1D profiles, 2D matrices, and 3D layered matrices"
    )


def _axis_index(value: Any, *, name: str, ndim: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer axis for a {ndim}D array")
    axis = int(value)
    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        raise ValueError(f"{name} must be between {-ndim} and {ndim - 1}")
    return axis


def _standard_layout(
    array: np.ndarray,
    data_type: DataType,
    *,
    time_axis: int | None,
    layer_axis: int | None,
) -> tuple[np.ndarray, int, int | None]:
    """把任意轴顺序转成 MSP 的标准布局。

    标准布局是 1-D ``(time,)``、2-D ``(time, feature)``、3-D
    ``(layer, time, feature)``。轴语义属于输入契约，不属于声化方法：预处理的
    基线是沿时间轴逐通道计算的，如果方法层才知道"其实时间是轴 1"，预处理就
    已经沿错误的轴扣过基线了，而且不会报错。
    """
    if data_type is DataType.PROFILE:
        if (
            isinstance(time_axis, (bool, np.bool_))
            or time_axis not in (None, 0)
            or layer_axis is not None
        ):
            raise ValueError("time_axis and layer_axis do not apply to 1D profiles")
        return array, 0, None

    if data_type is DataType.MATRIX:
        if layer_axis is not None:
            raise ValueError("layer_axis only applies to 3D layered matrices")
        resolved_time = 0 if time_axis is None else _axis_index(time_axis, name="time_axis", ndim=2)
        standardized = array if resolved_time == 0 else np.swapaxes(array, 0, 1)
        return standardized, resolved_time, None

    resolved_layer = 0 if layer_axis is None else _axis_index(layer_axis, name="layer_axis", ndim=3)
    if time_axis is None:
        # 未指定时，时间轴取第一个非层轴，与 (layer, time, feature) 的直觉一致。
        resolved_time = next(axis for axis in range(3) if axis != resolved_layer)
    else:
        resolved_time = _axis_index(time_axis, name="time_axis", ndim=3)
    if resolved_time == resolved_layer:
        raise ValueError("time_axis and layer_axis must refer to different axes")
    feature_axis = next(axis for axis in range(3) if axis not in (resolved_layer, resolved_time))
    return (
        np.transpose(array, (resolved_layer, resolved_time, feature_axis)),
        resolved_time,
        resolved_layer,
    )


@dataclass(frozen=True, eq=False)
class SonificationInput:
    """科学数组及其真实物理时长的不可变快照。

    构造时数组会被转成标准布局：1-D ``(time,)``、2-D ``(time, feature)``、
    3-D ``(layer, time, feature)``。``time_axis`` / ``layer_axis`` 随后描述该标准
    布局；``input_shape`` / ``source_time_axis`` / ``source_layer_axis`` 保留调用者
    原始数组的几何信息。之后的预处理和声化都只认标准布局。

    Equality and hashing use object identity. Comparing or hashing array
    contents implicitly would be ambiguous and unexpectedly expensive.

    Args:
        data: A 1-D profile, a 2-D matrix, or a 3-D stack of matrices.
        duration: Physical duration represented by the data, in seconds.
        data_type: Explicit type or ``None`` to infer it from dimensionality.
        name: Optional source label recorded in the result.
        time_axis: Which axis of ``data`` carries time. ``None`` uses axis 0
            for matrices and the first non-layer axis for layered matrices.
        layer_axis: Which axis of a 3-D array separates the layers.
            ``None`` uses axis 0.
    """

    data: np.ndarray
    duration: float
    data_type: DataType | str | None = None
    name: str | None = None
    time_axis: int | None = None
    layer_axis: int | None = None
    input_shape: tuple[int, ...] = field(init=False)
    source_time_axis: int = field(init=False)
    source_layer_axis: int | None = field(init=False)

    def __post_init__(self) -> None:
        duration = _positive_float(self.duration, name="duration")
        if self.name is None:
            name = None
        elif not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string or None")
        else:
            name = self.name.strip()

        # 先根据原始维数判断类型，再统一转换数值；这样形状错误的信息更直接。
        raw_array = np.asarray(self.data)
        resolved_type = (
            infer_data_type(raw_array)
            if self.data_type is None
            else parse_data_type(self.data_type)
        )
        expected_ndim = {
            DataType.PROFILE: 1,
            DataType.MATRIX: 2,
            DataType.LAYERED_MATRIX: 3,
        }[resolved_type]
        if raw_array.ndim != expected_ndim:
            raise ValueError(
                f"data_type='{resolved_type.value}' requires a {expected_ndim}D array, "
                f"got {raw_array.ndim}D"
            )
        # NaN 表示掩掉的通道或缺测样本，是真实的科学输入；是否接受由预处理的
        # nan_policy 决定。Inf 没有物理含义且会破坏分位数，任何策略下都拒绝。
        array = _as_real_array(raw_array, name="data", allow_nan=True)
        array, source_time_axis, source_layer_axis = _standard_layout(
            array,
            resolved_type,
            time_axis=self.time_axis,
            layer_axis=self.layer_axis,
        )
        # bytes-backed 数组同时完成复制和结构性冻结；调用者无法通过
        # ``setflags(write=True)`` 重新打开这个来源快照。
        array = _immutable_array(array, dtype=np.float64)

        object.__setattr__(self, "data", array)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "data_type", resolved_type)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "input_shape", tuple(raw_array.shape))
        object.__setattr__(self, "source_time_axis", source_time_axis)
        object.__setattr__(self, "source_layer_axis", source_layer_axis)
        # These fields describe the stored canonical ``data`` array. The
        # corresponding axes in the caller's original shape remain available
        # through source_time_axis/source_layer_axis.
        object.__setattr__(self, "time_axis", 0 if array.ndim < 3 else 1)
        object.__setattr__(self, "layer_axis", 0 if array.ndim == 3 else None)


__all__ = [
    "DataType",
    "SonificationInput",
    "infer_data_type",
    "parse_data_type",
]
