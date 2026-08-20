"""统一 API 的方法元数据、默认值和输入类型兼容关系。"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from ._perceptual_config import EVENT_DEFAULTS, PERCEPTUAL_DEFAULTS, VOICE_DEFAULTS
from .inputs import DataType, parse_data_type


@dataclass(frozen=True)
class MethodSpec:
    """一种主要声化方法的公开说明。

    ``input_feature_bins`` 描述共享预处理必须交付的输入形状。方法若在自己的
    checkpoint 适配器里完成固定尺寸映射（例如 HiFi-GAN 内部的 80 bin log-mel
    编码），该尺寸属于实现细节，由拥有该 checkpoint 的方法自行保存。
    """

    name: str
    input_types: tuple[DataType, ...]
    defaults: Mapping[str, Any]
    description: str
    runner_module: str
    runner_name: str
    synthesizes_duration: bool = False
    default_repeat: int = 1
    default_for: tuple[DataType, ...] = ()
    optional_extra: str | None = None
    # 方法要求共享预处理交付的固定特征轴格数；预处理据此设置 feature_rebin。
    # 若方法在自己的 checkpoint 适配器中完成固定尺寸映射，则这里必须为 None。
    input_feature_bins: int | None = None
    # 方法的输出帧几何 ``(module, function)``；函数接受已解析的 method_params，
    # 返回 ``(sample_rate, hop_length)``，供 time_rebin='auto' 反推输入帧数。
    frame_geometry: tuple[str, str] | None = None
    # 可变特征几何 ``(module, function)``；函数返回 ``(default_bins,
    # max_bins)``。它适用于上限由参数决定的方法，例如 n_fft 改变时的 Griffin-Lim。
    feature_geometry: tuple[str, str] | None = None
    # 未显式指定 time_rebin 时使用的默认值（目前只有 'auto' 或 None）。
    default_time_rebin: str | None = None
    # time_rebin='auto' 是否允许把时间轴插值到比源数据更多的帧。
    # 对纯信号处理的方法这是划算的：不这样做就只能在合成之后拉伸波形，
    # 而那会整体移动音高。对神经声码器则关闭，避免给模型喂插值出来的帧。
    allow_frame_upsampling: bool = False
    # 相邻重复段共享的边界帧数。Griffin--Lim 的 N 帧 ISTFT 产生
    # ``(N - 1) * hop`` 个样本，所以每个后续副本复用一个边界帧。
    repeat_frame_overlap: int = 0
    # 分组扩展参数的默认值 ``{group: defaults}``。这些组在 ``defaults`` 里以
    # ``None`` 出现（表示"整组用默认值"），所以发现入口只能看到组名本身。把组
    # 一并注册，``list-settings`` 和 ``available_methods()`` 才能展开组内真正
    # 可调的键，读源码之外也能查到 ``detune_cents`` 这类设置。
    grouped_defaults: Mapping[str, Mapping[str, Any]] | None = None
    output_channels: int = 1
    output_peak: float | None = 0.9
    # 方法内部若存在数据依赖的量（如 HiFi-GAN 的直方图偏移），通过 ``provenance``
    # 出参交回统一 API 记入结果，避免出现结果里看不出来的隐藏变换。
    emits_provenance: bool = False

    @property
    def parameters(self) -> tuple[str, ...]:
        """该方法允许出现在 ``method_params`` 中的参数名。"""
        return tuple(self.defaults)

    def supports(self, data_type: DataType | str) -> bool:
        """该方法是否接受指定科学数据类型。"""
        return parse_data_type(data_type) in self.input_types

    def load_runner(self) -> Callable[..., tuple[Any, int]]:
        """按注册路径加载实现，保持可选后端惰性。"""
        return cast(
            Callable[..., tuple[Any, int]],
            _load_callable(self.runner_module, self.runner_name),
        )

    def resolve_frame_geometry(self, method_params: Mapping[str, Any]) -> tuple[int, int]:
        """返回该方法在给定参数下的 ``(sample_rate, hop_length)``。"""
        if self.frame_geometry is None:
            raise RuntimeError(
                f"method '{self.name}' does not register frame geometry; "
                "time_rebin='auto' is unavailable"
            )
        resolver = cast(
            Callable[[Mapping[str, Any]], tuple[int, int]],
            _load_callable(*self.frame_geometry),
        )
        sample_rate, hop_length = resolver(method_params)
        return int(sample_rate), int(hop_length)

    def resolve_feature_geometry(self, method_params: Mapping[str, Any]) -> tuple[int, int]:
        """返回给定参数下的 ``(default_bins, max_bins)``。"""
        if self.feature_geometry is None:
            raise RuntimeError(f"method '{self.name}' does not register feature geometry")
        resolver = cast(
            Callable[[Mapping[str, Any]], tuple[int, int]],
            _load_callable(*self.feature_geometry),
        )
        default_bins, max_bins = resolver(method_params)
        return int(default_bins), int(max_bins)


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
    preflight_module: str | None = None
    preflight_name: str | None = None
    # 后处理器能接受的最大声道数。统一 API 在主声化之前就用它拒绝不兼容组合，
    # 而不是让立体声结果跑完整段合成后才在后处理入口失败。
    max_input_channels: int = 2

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(self.defaults)

    def load_runner(self) -> Callable[..., tuple[Any, int]]:
        return cast(
            Callable[..., tuple[Any, int]],
            _load_callable(self.runner_module, self.runner_name),
        )

    def validate_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """用注册的验证器规范化参数；无验证器时返回普通副本。"""
        if self.validator_module is None or self.validator_name is None:
            return dict(params)
        validator = cast(
            Callable[..., dict[str, Any]],
            _load_callable(self.validator_module, self.validator_name),
        )
        return validator(**params)

    def preflight(self, params: Mapping[str, Any]) -> None:
        """Check optional runtime and asset availability before primary synthesis."""
        if self.preflight_module is None and self.preflight_name is None:
            return
        if self.preflight_module is None or self.preflight_name is None:
            raise RuntimeError(
                f"postprocessor '{self.name}' has an incomplete preflight registration"
            )
        checker = cast(
            Callable[..., None],
            _load_callable(self.preflight_module, self.preflight_name),
        )
        checker(**params)


def _defaults(**values: Any) -> Mapping[str, Any]:
    """冻结默认参数，避免注册表在运行期间被意外修改。"""
    return MappingProxyType(dict(values))


def _load_callable(module_name: str, runner_name: str) -> Callable[..., Any]:
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
        input_types=(DataType.PROFILE, DataType.MATRIX),
        defaults=_defaults(
            sr=48_000,
            instrument=None,
        ),
        description="Interpolate the time profile, optionally convolving a synthesized response.",
        runner_module="radiosonify.profile",
        runner_name="profile_to_wave",
        synthesizes_duration=True,
        # 二维输入沿特征轴平均成轮廓。用 feature_rebin=1 表达，这样"求平均"
        # 也走统一预处理，并出现在结果的 preprocess_params 里。
        input_feature_bins=1,
    ),
    MethodSpec(
        name="amplitude",
        input_types=(DataType.PROFILE, DataType.MATRIX),
        defaults=_defaults(
            sr=48_000,
            freq=1_000.0,
            compression=0.0,
            harmonics=4,
            harmonic_decay=1.0,
        ),
        description="Map a profile linearly to a fixed harmonic carrier's amplitude envelope.",
        runner_module="radiosonify.amplitude",
        runner_name="amplitude_modulate",
        synthesizes_duration=True,
        default_repeat=5,
        default_for=(DataType.PROFILE,),
        input_feature_bins=1,
    ),
    MethodSpec(
        name="erb",
        input_types=(DataType.MATRIX,),
        defaults=_defaults(**PERCEPTUAL_DEFAULTS),
        description=(
            "Map time and feature position to time and perceptual pitch, with "
            "low-level brightness ambience and continuous temporal detail."
        ),
        runner_module="radiosonify.erb",
        runner_name="erb_sonify",
        synthesizes_duration=True,
        default_for=(DataType.MATRIX,),
        grouped_defaults=_defaults(
            voice_params=VOICE_DEFAULTS,
            event_params=EVENT_DEFAULTS,
        ),
        output_peak=None,
    ),
    MethodSpec(
        name="griffinlim",
        input_types=(DataType.MATRIX,),
        defaults=_defaults(
            sr=48_000,
            n_iter=64,
            n_fft=4096,
            frame_length=0.04,
            preemphasis=0.0,
            max_db=100.0,
            ref_db=20.0,
        ),
        description="Treat the full 2-D intensity array as a mel-like magnitude map.",
        runner_module="radiosonify.griffinlim",
        runner_name="griffinlim",
        # Griffin--Lim 的输出长度完全由输入帧数决定，因此和 HiFi-GAN 一样支持
        # 由目标时长反推输入帧数。不这样做就只能先合成再重采样：实测
        # speed=0.5 时会出现 6.3 倍的多相拉伸，音高整体下移、带宽塌陷。
        frame_geometry=("radiosonify.griffinlim", "_frame_geometry"),
        feature_geometry=("radiosonify.griffinlim", "_feature_geometry"),
        default_time_rebin="auto",
        allow_frame_upsampling=True,
        repeat_frame_overlap=1,
    ),
    MethodSpec(
        name="hifigan",
        input_types=(DataType.MATRIX,),
        defaults=_defaults(),
        description=(
            "Adapt the full 2-D array to the pretrained neural vocoder, choosing "
            "time frames from the requested audio duration by default."
        ),
        runner_module="radiosonify.hifigan",
        runner_name="hifigan",
        optional_extra="hifigan",
        frame_geometry=("radiosonify.hifigan", "_frame_geometry"),
        default_time_rebin="auto",
        output_peak=None,
        emits_provenance=True,
    ),
    MethodSpec(
        name="spatial_erb",
        input_types=(DataType.LAYERED_MATRIX,),
        defaults=_defaults(
            **PERCEPTUAL_DEFAULTS,
            pan_positions=None,
            layer_gains=None,
        ),
        description=(
            "Apply the same continuous perceptual mapping to 3-D layers and pan "
            "the resulting voices across a stereo field."
        ),
        runner_module="radiosonify.spatial",
        runner_name="spatial_sonify",
        synthesizes_duration=True,
        default_for=(DataType.LAYERED_MATRIX,),
        grouped_defaults=_defaults(
            voice_params=VOICE_DEFAULTS,
            event_params=EVENT_DEFAULTS,
        ),
        output_channels=2,
        output_peak=None,
    ),
)

_METHOD_BY_NAME = {spec.name: spec for spec in _METHODS}
_METHOD_ALIASES = {
    "profile_to_wave": "profile",
    "amplitude_modulate": "amplitude",
    "erb_filterbank": "erb",
    "griffin_lim": "griffinlim",
    "spatial": "spatial_erb",
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
        preflight_module="radiosonify.musicnet",
        preflight_name="_preflight_musicnet",
        # 预训练编码器的输入契约是单声道；三维空间化的立体声结果无法直接送入。
        max_input_channels=1,
    ),
    PostprocessorSpec(
        name="rave",
        label="RAVE",
        defaults=_defaults(
            model_path=None,
            device="auto",
        ),
        description="Apply a user-supplied exported RAVE model as an aesthetic timbre transform.",
        runner_module="radiosonify.rave",
        runner_name="rave",
        optional_extra="rave",
        validator_module="radiosonify.rave",
        validator_name="_validate_rave_parameters",
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
