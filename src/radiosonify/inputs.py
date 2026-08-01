"""统一声化 API 的科学数据类型与不可变输入快照。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .core import _as_finite_array, _positive_float


class DataType(str, Enum):
    """MSP 接受的两类科学数据。"""

    PROFILE = "profile"
    DYNAMIC_SPECTRUM = "dynamic_spectrum"

    def __str__(self) -> str:
        return self.value


_DATA_TYPE_ALIASES = {
    "profile": DataType.PROFILE,
    "pulse_profile": DataType.PROFILE,
    "dynamic_spectrum": DataType.DYNAMIC_SPECTRUM,
    "dynamic_spectra": DataType.DYNAMIC_SPECTRUM,
    "spectrogram": DataType.DYNAMIC_SPECTRUM,
}


def parse_data_type(value: DataType | str) -> DataType:
    """把公开字符串及别名解析为标准 :class:`DataType`。"""
    if isinstance(value, DataType):
        return value
    if not isinstance(value, str):
        raise ValueError("data_type must be 'profile' or 'dynamic_spectrum'")
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _DATA_TYPE_ALIASES[key]
    except KeyError as exc:
        raise ValueError("data_type must be 'profile' or 'dynamic_spectrum'") from exc


def infer_data_type(data: np.ndarray) -> DataType:
    """一维推断为轮廓，二维推断为动态谱。"""
    array = np.asarray(data)
    if array.ndim == 1:
        return DataType.PROFILE
    if array.ndim == 2:
        return DataType.DYNAMIC_SPECTRUM
    raise ValueError(
        f"cannot infer data_type from a {array.ndim}D array; "
        "MSP accepts only 1D profiles and 2D dynamic spectra"
    )


@dataclass(frozen=True, eq=False)
class SonificationInput:
    """科学数组及其真实物理时长的不可变快照。

    Equality and hashing use object identity. Comparing or hashing array
    contents implicitly would be ambiguous and unexpectedly expensive.

    Args:
        data: A 1-D profile or 2-D time x frequency dynamic spectrum.
        duration: Physical duration represented by the data, in seconds.
        data_type: Explicit type or ``None`` to infer it from dimensionality.
        name: Optional source label recorded in the result.
    """

    data: np.ndarray
    duration: float
    data_type: DataType | str | None = None
    name: str | None = None

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
        expected_ndim = 1 if resolved_type is DataType.PROFILE else 2
        if raw_array.ndim != expected_ndim:
            raise ValueError(
                f"data_type='{resolved_type.value}' requires a {expected_ndim}D array, "
                f"got {raw_array.ndim}D"
            )
        array = _as_finite_array(raw_array, name="data").copy()
        # frozen dataclass 只冻结属性绑定；把数组设为只读才能真正防止来源在声化途中改变。
        array.setflags(write=False)

        object.__setattr__(self, "data", array)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "data_type", resolved_type)
        object.__setattr__(self, "name", name)


__all__ = [
    "DataType",
    "SonificationInput",
    "infer_data_type",
    "parse_data_type",
]
