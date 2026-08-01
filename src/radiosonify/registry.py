"""统一 API 的方法元数据、默认值和输入类型兼容关系。"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .inputs import DataType, parse_data_type


@dataclass(frozen=True)
class MethodSpec:
    """一种主要声化方法的公开说明。"""

    name: str
    label: str
    input_types: tuple[DataType, ...]
    defaults: Mapping[str, Any]
    description: str
    runner_module: str
    runner_name: str
    synthesizes_duration: bool = False
    supports_repeat: bool = False
    default_for: tuple[DataType, ...] = ()
    optional_extra: str | None = None

    @property
    def parameters(self) -> tuple[str, ...]:
        """该方法允许出现在 ``method_params`` 中的参数名。"""
        return tuple(self.defaults)

    def supports(self, data_type: DataType | str) -> bool:
        """该方法是否接受指定科学数据类型。"""
        return parse_data_type(data_type) in self.input_types

    def load_runner(self) -> Callable[..., tuple[Any, int]]:
        """按注册路径加载实现，保持可选后端惰性。"""
        return _load_runner(self.runner_module, self.runner_name)


@dataclass(frozen=True)
class PostprocessorSpec:
    """一种音频后处理器的公开说明与运行入口。"""

    name: str
    label: str
    defaults: Mapping[str, Any]
    description: str
    runner_module: str
    runner_name: str
    optional_extra: str | None = None
    validator_module: str | None = None
    validator_name: str | None = None

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(self.defaults)

    def load_runner(self) -> Callable[..., tuple[Any, int]]:
        return _load_runner(self.runner_module, self.runner_name)

    def validate_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """用注册的验证器规范化参数；无验证器时返回普通副本。"""
        if self.validator_module is None or self.validator_name is None:
            return dict(params)
        validator = _load_runner(self.validator_module, self.validator_name)
        return validator(**params)


def _defaults(**values: Any) -> Mapping[str, Any]:
    """冻结默认参数，避免注册表在运行期间被意外修改。"""
    return MappingProxyType(dict(values))


def _load_runner(module_name: str, runner_name: str) -> Callable[..., tuple[Any, int]]:
    """加载并检查注册的函数，给注册表漂移提供直接诊断。"""
    try:
        runner = getattr(importlib.import_module(module_name), runner_name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"registered runner '{module_name}.{runner_name}' is unavailable"
        ) from exc
    if not callable(runner):
        raise RuntimeError(f"registered runner '{module_name}.{runner_name}' is not callable")
    return runner


# 方法按由轻量、可解释到神经模型的顺序排列，展示顺序和自动选择都由此表决定。
_METHODS = (
    MethodSpec(
        name="profile",
        label="Profile mapping",
        input_types=(DataType.PROFILE, DataType.DYNAMIC_SPECTRUM),
        defaults=_defaults(
            sr=48_000,
            time_downsample=None,
            instrument=None,
        ),
        description="Interpolate the time profile, optionally convolving a synthesized response.",
        runner_module="radiosonify.profile",
        runner_name="profile_to_wave",
        synthesizes_duration=True,
        supports_repeat=True,
    ),
    MethodSpec(
        name="amplitude",
        label="Amplitude modulation",
        input_types=(DataType.PROFILE, DataType.DYNAMIC_SPECTRUM),
        defaults=_defaults(
            sr=48_000,
            freq=1_000.0,
            compression=99.0,
            time_downsample=None,
        ),
        description="Map the time profile to the amplitude envelope of a sine carrier.",
        runner_module="radiosonify.amplitude",
        runner_name="amplitude_modulate",
        synthesizes_duration=True,
        supports_repeat=True,
        default_for=(DataType.PROFILE,),
    ),
    MethodSpec(
        name="griffinlim",
        label="Griffin-Lim reconstruction",
        input_types=(DataType.DYNAMIC_SPECTRUM,),
        defaults=_defaults(
            sr=48_000,
            n_iter=64,
            n_fft=4096,
            frame_length=0.04,
            preemphasis=0.0,
            max_db=100.0,
            ref_db=20.0,
            time_rebin=None,
            freq_rebin=None,
            clean=False,
            exposure_cut=25,
        ),
        description="Treat the full 2-D intensity array as a mel-like magnitude map.",
        runner_module="radiosonify.griffinlim",
        runner_name="griffinlim",
        default_for=(DataType.DYNAMIC_SPECTRUM,),
    ),
    MethodSpec(
        name="hifigan",
        label="HiFi-GAN vocoder",
        input_types=(DataType.DYNAMIC_SPECTRUM,),
        defaults=_defaults(
            time_rebin=None,
            time_smoothing=None,
            clean=False,
            exposure_cut=25,
        ),
        description="Resize the full 2-D array to the pretrained neural vocoder input.",
        runner_module="radiosonify.hifigan",
        runner_name="hifigan",
        optional_extra="hifigan",
    ),
)

_METHOD_BY_NAME = {spec.name: spec for spec in _METHODS}
_METHOD_ALIASES = {
    "profile_to_wave": "profile",
    "amplitude_modulate": "amplitude",
    "griffin_lim": "griffinlim",
}

_POSTPROCESSORS = (
    PostprocessorSpec(
        name="musicnet",
        label="MusicNet",
        defaults=_defaults(
            decoder_id=2,
            checkpoint_type="bestmodel",
            split_size=20,
            num_threads=1,
            seed=0,
        ),
        description="Apply the pretrained MusicNet/WaveNet audio style translator.",
        runner_module="radiosonify.musicnet",
        runner_name="musicnet",
        optional_extra="musicnet",
        validator_module="radiosonify.musicnet",
        validator_name="_validate_musicnet_parameters",
    ),
)
_POSTPROCESSOR_BY_NAME = {spec.name: spec for spec in _POSTPROCESSORS}


def available_methods(data_type: DataType | str | None = None) -> tuple[MethodSpec, ...]:
    """列出主要方法；可按科学输入类型过滤。"""
    if data_type is None:
        return _METHODS
    resolved = parse_data_type(data_type)
    return tuple(spec for spec in _METHODS if spec.supports(resolved))


def available_postprocessors() -> tuple[PostprocessorSpec, ...]:
    """列出可用于统一 API 的音频后处理器。"""
    return _POSTPROCESSORS


def default_method(data_type: DataType | str) -> str:
    """返回指定输入类型的不依赖神经权重的默认方法。"""
    resolved = parse_data_type(data_type)
    for spec in _METHODS:
        if resolved in spec.default_for:
            return spec.name
    raise RuntimeError(f"no default sonification method registered for {resolved.value}")


def resolve_method(method: str, data_type: DataType | str) -> MethodSpec:
    """解析 ``auto``/别名，并检查方法与数据类型是否兼容。"""
    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a non-empty string")
    resolved_type = parse_data_type(data_type)
    key = method.strip().lower().replace("-", "_").replace(" ", "_")
    if key == "auto":
        key = default_method(resolved_type)
    key = _METHOD_ALIASES.get(key, key)

    try:
        spec = _METHOD_BY_NAME[key]
    except KeyError as exc:
        choices = ", ".join(item.name for item in available_methods(resolved_type))
        raise ValueError(
            f"unknown method '{method}' for {resolved_type.value}; available: {choices}"
        ) from exc
    if not spec.supports(resolved_type):
        choices = ", ".join(item.name for item in available_methods(resolved_type))
        raise ValueError(
            f"method '{spec.name}' does not accept {resolved_type.value}; available: {choices}"
        )
    return spec


def resolve_postprocessor(name: str) -> PostprocessorSpec:
    """解析后处理器名称，并给未知值列出可用选择。"""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("postprocess must be a non-empty string")
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _POSTPROCESSOR_BY_NAME[key]
    except KeyError as exc:
        choices = ", ".join(spec.name for spec in _POSTPROCESSORS)
        raise ValueError(f"unknown postprocess '{name}'; available: {choices}") from exc


__all__ = [
    "MethodSpec",
    "PostprocessorSpec",
    "available_methods",
    "available_postprocessors",
    "default_method",
    "resolve_method",
    "resolve_postprocessor",
]
