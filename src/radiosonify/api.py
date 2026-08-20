"""按数据类型选择方法、控制时长并记录来源的统一声化流程。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .core import (
    _as_finite_array,
    _boolean,
    _immutable_array,
    _merge_settings,
    _positive_int,
    _wav_output_path,
    save_audio,
)
from .inputs import DataType, SonificationInput
from .preprocessing import _preprocess_validated, resolve_preprocess_params
from .registry import MethodSpec, PostprocessorSpec, resolve_method, resolve_postprocessor
from .timing import (
    _resample_audio_rate,
    condition_audio_output,
    duration_to_frames,
    duration_to_samples,
    fit_audio_duration,
    target_audio_duration,
)


@dataclass(frozen=True, eq=False)
class SonificationResult:
    """最终音频及复现本次统一转换所需的来源信息。

    Equality and hashing use object identity rather than implicitly comparing
    or hashing potentially large NumPy arrays.
    """

    audio: np.ndarray
    sample_rate: int
    data_type: DataType
    data_duration: float
    input_shape: tuple[int, ...]
    source_time_axis: int
    source_layer_axis: int | None
    method: str
    preprocess_params: Mapping[str, Any]
    method_params: Mapping[str, Any]
    speed: float
    repeat: int
    preserve_pitch: bool
    target_duration: float
    output_duration: float
    method_sample_rate: int
    method_native_samples: int
    method_native_duration: float
    method_time_scale: float
    source_name: str | None = None
    postprocess: str | None = None
    postprocess_params: Mapping[str, Any] | None = None
    postprocess_native_samples: int | None = None
    postprocess_native_duration: float | None = None
    postprocess_time_scale: float | None = None
    output_path: Path | None = None


# ---------- 输入和参数解析 ----------


def _coerce_source(
    source: SonificationInput | np.ndarray,
    *,
    data_duration: float | None,
    data_type: DataType | str | None,
) -> SonificationInput:
    if isinstance(source, SonificationInput):
        if data_duration is not None or data_type is not None:
            raise ValueError(
                "data_duration and data_type must be stored in SonificationInput "
                "when source is already a SonificationInput"
            )
        return source
    if data_duration is None:
        raise ValueError(
            "data_duration is required when source is an array; "
            "it is the physical time span represented by the data"
        )
    return SonificationInput(
        data=np.asarray(source),
        duration=data_duration,
        data_type=data_type,
    )


def _freeze_provenance_value(value: Any) -> Any:
    """递归复制并冻结参数值，避免结果元数据受调用者后续修改。

    分支按抽象基类而不是具体类型判断。方法参数的公开契约接受任意
    ``Sequence``（例如 ``spatial_erb`` 的 ``pan_positions``），只认 ``list`` /
    ``tuple`` 会把 ``UserList`` 这类自定义序列按引用存进结果：实测调用者随后
    改动原对象，``result.method_params`` 会跟着变，与"冻结结果"的承诺矛盾。
    """
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_provenance_value(item) for key, item in value.items()}
        )
    if isinstance(value, str):
        # 文本本身就是不可变序列，且逐字符递归会把 'mad' 这样的选项变成字符元组。
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return _freeze_provenance_value(value.tolist())
        return _immutable_array(value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_provenance_value(item) for item in value)
    if isinstance(value, Sequence):
        return tuple(_freeze_provenance_value(item) for item in value)
    return value


def _freeze_parameters(params: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_provenance_value(value) for key, value in params.items()})


_MOVED_TO_PREPROCESSING = {
    "n_mels": "feature_rebin",
    "freq_rebin": "feature_rebin",
    "time_rebin": "time_rebin",
    "time_smoothing": "time_smoothing",
    "time_downsample": "time_rebin",
    "time_axis": "SonificationInput(time_axis=...)",
    "layer_axis": "SonificationInput(layer_axis=...)",
}


def _effective_params(
    spec: MethodSpec,
    supplied: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(supplied, Mapping):
        moved = sorted(set(supplied) & set(_MOVED_TO_PREPROCESSING) - set(spec.defaults))
        if moved:
            # 这些旋钮曾经在方法层改数据。给出精确的迁移目标，比笼统的
            # "unknown parameter" 更省事，因为它们的语义并没有消失。
            details = "; ".join(f"{name} -> {_MOVED_TO_PREPROCESSING[name]}" for name in moved)
            raise ValueError(
                f"method_params {moved} no longer belong to method '{spec.name}'; "
                f"all data-domain settings live in preprocess_params: {details}"
            )
    params = _merge_settings(
        spec.defaults,
        supplied,
        field_name="method_params",
        unknown_label=f"unknown parameter(s) for method '{spec.name}'",
    )
    # 分组参数在 defaults 里只是一个 None 占位。把注册的组默认值就地展开，
    # 溯源里分组和顶层参数才是同一口径（顶层 19 个键全部记录，分组也应如此），
    # 组内拼错的键也能在合成开始之前就被拒绝。
    for group, group_defaults in (spec.grouped_defaults or {}).items():
        params[group] = _merge_settings(
            group_defaults,
            params[group],
            field_name=group,
            unknown_label=f"unknown {group} key(s)",
        )
    return params


def _effective_postprocess(
    postprocess: str | None,
    supplied: Mapping[str, Any] | None,
) -> tuple[PostprocessorSpec | None, dict[str, Any] | None]:
    """解析后处理并在主要声化开始前完成全部参数校验。"""
    if postprocess is None:
        if supplied:
            raise ValueError("postprocess_params requires postprocess to be selected")
        return None, None
    spec = resolve_postprocessor(postprocess)

    params = _merge_settings(
        spec.defaults,
        supplied,
        field_name="postprocess_params",
        unknown_label=f"unknown {spec.label} postprocess parameter(s)",
    )
    return spec, spec.validate_params(params)


# ---------- 方法执行 ----------


def _run_primary(
    data: np.ndarray,
    spec: MethodSpec,
    target_duration: float,
    params: dict[str, Any],
    provenance: dict[str, Any],
) -> tuple[np.ndarray, int]:
    runner = spec.load_runner()
    controls: dict[str, Any] = {"output": None}
    if spec.synthesizes_duration:
        controls["duration"] = target_duration
    if spec.emits_provenance:
        controls["provenance"] = provenance
    return runner(data, **controls, **params)


def _run_postprocess(
    audio: np.ndarray,
    sr: int,
    postprocess: PostprocessorSpec | None,
    params: Mapping[str, Any] | None,
) -> tuple[np.ndarray, int]:
    if postprocess is None:
        return audio, sr
    if params is None:
        raise RuntimeError("postprocess parameters were not resolved")
    processed, output_sr = postprocess.load_runner()(audio, sr=sr, output=None, **params)
    return processed, output_sr


def _apply_method_geometry(
    preprocess_params: dict[str, Any],
    spec: MethodSpec,
    method_params: Mapping[str, Any],
    *,
    source_ndim: int,
    source_time_bins: int,
    primary_duration: float,
    repeat: int,
) -> dict[str, Any]:
    """把方法声明的输入几何解析成具体的预处理参数。

    固定或参数相关的特征几何只描述共享预处理交付的尺寸；checkpoint 内部的
    固定编码尺寸（例如 HiFi-GAN 的 80 bin）不在这里解析。时间帧率仍用于把
    ``time_rebin='auto'`` 解析成共享预处理的具体帧数。
    """
    params = dict(preprocess_params)

    # 一维轮廓没有特征轴。像 profile/amplitude 这样同时接受 1-D 和 2-D 的方法，
    # 只有在拿到 2-D 输入时才需要把特征轴压成一条轮廓。
    if source_ndim > 1:
        requested_feature_bins = params["feature_rebin"]
        if spec.feature_geometry is not None:
            default_bins, max_bins = spec.resolve_feature_geometry(method_params)
            default_bins = _positive_int(default_bins, name="registered default feature bins")
            max_bins = _positive_int(max_bins, name="registered maximum feature bins")
            if default_bins > max_bins:
                raise RuntimeError(
                    f"method '{spec.name}' registers default feature bins above its maximum"
                )
            if requested_feature_bins in (None, "auto"):
                requested_feature_bins = default_bins
            elif requested_feature_bins > max_bins:
                raise ValueError(
                    f"feature_rebin ({requested_feature_bins}) cannot exceed {max_bins} "
                    f"for method '{spec.name}' with the current method_params"
                )
        elif spec.input_feature_bins is not None:
            required_bins = _positive_int(
                spec.input_feature_bins,
                name="registered input_feature_bins",
            )
            if requested_feature_bins in (None, "auto"):
                requested_feature_bins = required_bins
            elif requested_feature_bins != required_bins:
                raise ValueError(
                    f"method '{spec.name}' requires feature_rebin={required_bins}, "
                    f"got {requested_feature_bins}"
                )
        elif requested_feature_bins == "auto":
            raise ValueError(
                f"feature_rebin='auto' requires feature geometry, which method "
                f"'{spec.name}' does not register"
            )
        params["feature_rebin"] = requested_feature_bins

    requested = params["time_rebin"]
    if requested is None and spec.default_time_rebin is not None:
        requested = spec.default_time_rebin
    if requested == "auto":
        if spec.frame_geometry is None:
            raise ValueError(
                f"time_rebin='auto' requires frame geometry, which method "
                f"'{spec.name}' does not register"
            )
        sample_rate, hop_length = spec.resolve_frame_geometry(method_params)
        # ``repeat`` 是在预处理末尾沿时间轴拼接的，所以这里要算的是**单遍**
        # 需要多少帧，拼接 repeat 次之后才对得上目标时长。
        automatic_bins = (
            duration_to_frames(
                primary_duration / repeat,
                sample_rate,
                hop_length,
            )
            + spec.repeat_frame_overlap
        )
        requested = (
            automatic_bins if spec.allow_frame_upsampling else min(automatic_bins, source_time_bins)
        )
    params["time_rebin"] = requested
    return params


# ---------- 公开统一入口 ----------


def sonify(
    source: SonificationInput | np.ndarray,
    *,
    data_duration: float | None = None,
    data_type: DataType | str | None = None,
    method: str = "auto",
    speed: float = 1.0,
    repeat: int | None = None,
    preserve_pitch: bool = False,
    output_sr: int | None = None,
    preprocess_params: Mapping[str, Any] | None = None,
    method_params: Mapping[str, Any] | None = None,
    postprocess: str | None = None,
    postprocess_params: Mapping[str, Any] | None = None,
    output: str | Path | None = None,
) -> SonificationResult:
    """通过统一流程声化一维轮廓、二维矩阵或三维分层矩阵。

    Arrays require ``data_duration``. One-dimensional arrays are inferred as
    profiles, two-dimensional arrays as matrices, and three-dimensional arrays
    as layered matrices unless ``data_type`` is explicitly supplied. The final
    sample count is ``round(sample_rate * data_duration * repeat / speed)``.
    ``repeat`` works for every dimensionality and method because it is applied
    to the data during preprocessing rather than by an individual method. When
    omitted, each method's registered default is used (five for amplitude, one
    otherwise).

    Scientific-data conditioning happens in one place: the shared preprocessing
    stage rebins, baseline-corrects, optionally equalizes per-channel noise,
    clips, smooths, normalizes to ``[0, 1]`` and repeats, in that fixed order.
    A method may then apply a fixed checkpoint encoding (for example HiFi-GAN's
    internal 80-bin log-mel adapter); that encoding has no user-facing data
    controls. ``output_sr`` optionally converts the final container sample rate
    without synthesizing bandwidth above the method's native Nyquist frequency.
    """
    validated_source = _coerce_source(
        source,
        data_duration=data_duration,
        data_type=data_type,
    )
    resolved_data_type = cast(DataType, validated_source.data_type)
    preserve_pitch = _boolean(preserve_pitch, name="preserve_pitch")
    if output_sr is not None:
        output_sr = _positive_int(output_sr, name="output_sr")
    # 输出路径也必须在耗时合成前失败，避免最后一步才发现扩展名或目录错误。
    output_path = None if output is None else _wav_output_path(output)

    spec = resolve_method(method, resolved_data_type)
    repeat = spec.default_repeat if repeat is None else _positive_int(repeat, name="repeat")
    params = _effective_params(spec, method_params)
    requested_preprocess_params = resolve_preprocess_params(
        resolved_data_type,
        preprocess_params,
    )
    postprocess_spec, effective_postprocess_params = _effective_postprocess(
        postprocess,
        postprocess_params,
    )
    if postprocess_spec is not None and spec.output_channels > postprocess_spec.max_input_channels:
        # 在合成之前拒绝不兼容的组合。三维空间化的立体声送不进单声道的
        # MusicNet 编码器，而那段合成可能要跑几分钟。
        raise ValueError(
            f"postprocess '{postprocess_spec.name}' accepts at most "
            f"{postprocess_spec.max_input_channels} channel(s), but method "
            f"'{spec.name}' produces {spec.output_channels}"
        )
    if postprocess_spec is not None:
        if effective_postprocess_params is None:
            raise RuntimeError("postprocess parameters were not resolved")
        postprocess_spec.preflight(effective_postprocess_params)
    requested_duration = target_audio_duration(
        validated_source.duration,
        speed,
        repeat,
    )
    if output_sr is not None:
        # 已知最终采样率时在耗时合成前检查样本数是否可表示。
        try:
            duration_to_samples(requested_duration, output_sr)
        except ValueError as exc:
            raise ValueError(f"output_sr cannot represent the requested duration: {exc}") from exc
    # MusicNet 在模型原生时间尺度上运行；speed 只在生成后作为播放速度应用，
    # 避免 speed < 1 先把逐样本自回归工作量成倍放大。
    primary_duration = (
        target_audio_duration(validated_source.duration, 1.0, repeat)
        if postprocess_spec is not None
        else requested_duration
    )

    # 无后处理时直接拟合目标时长；有后处理时先拟合正常播放时长。
    source_time_bins = validated_source.data.shape[0 if validated_source.data.ndim < 3 else 1]
    effective_preprocess_params = _apply_method_geometry(
        requested_preprocess_params,
        spec,
        params,
        source_ndim=validated_source.data.ndim,
        source_time_bins=source_time_bins,
        primary_duration=primary_duration,
        repeat=repeat,
    )
    preprocessed_data = _preprocess_validated(
        validated_source.data,
        effective_preprocess_params,
        repeat=repeat,
        repeat_overlap=spec.repeat_frame_overlap,
    )
    preprocessed_data.setflags(write=False)
    method_provenance: dict[str, Any] = {}
    audio, sr = _run_primary(
        preprocessed_data,
        spec,
        primary_duration,
        params,
        method_provenance,
    )
    sr = _positive_int(sr, name="method sample rate")
    method_native_samples = len(audio)
    method_native_duration = method_native_samples / sr
    audio = fit_audio_duration(
        audio,
        sr,
        primary_duration,
        preserve_pitch=preserve_pitch,
    )
    method_time_scale = len(audio) / method_native_samples

    final_sr = sr
    postprocess_native_samples = None
    postprocess_native_duration = None
    postprocess_time_scale = None
    if postprocess_spec is not None:
        audio, final_sr = _run_postprocess(
            audio,
            sr,
            postprocess_spec,
            effective_postprocess_params,
        )
        final_sr = _positive_int(final_sr, name="postprocess sample rate")
        postprocess_native_samples = len(audio)
        postprocess_native_duration = postprocess_native_samples / final_sr
        audio = fit_audio_duration(
            _as_finite_array(audio, name="postprocessed audio", ndim=(1, 2)),
            final_sr,
            requested_duration,
            preserve_pitch=preserve_pitch,
        )
        postprocess_time_scale = len(audio) / postprocess_native_samples

    if output_sr is not None and output_sr != final_sr:
        # 这里只改变容器采样率，不改变物理时长或音高。相位声码器按旧采样率
        # 拉长后再用新采样率播放会把音高抬高 output_sr / final_sr 倍，
        # 因此采样率转换必须固定使用多相路径。
        audio = _resample_audio_rate(
            audio,
            final_sr,
            output_sr,
            target_samples=duration_to_samples(requested_duration, output_sr),
        )
        final_sr = output_sr
    output_peak = spec.output_peak if postprocess_spec is None else 0.9
    audio = condition_audio_output(audio, final_sr, peak=output_peak)

    if output_path is not None:
        save_audio(audio, final_sr, output_path)
    # 与 SonificationInput 一致，使用不可变底层缓冲区保证结果内容可追溯。
    audio = _immutable_array(audio)

    return SonificationResult(
        audio=audio,
        sample_rate=final_sr,
        data_type=resolved_data_type,
        data_duration=validated_source.duration,
        input_shape=validated_source.input_shape,
        source_time_axis=validated_source.source_time_axis,
        source_layer_axis=validated_source.source_layer_axis,
        method=spec.name,
        preprocess_params=_freeze_parameters(effective_preprocess_params),
        method_params=_freeze_parameters({**params, **method_provenance}),
        speed=float(speed),
        repeat=repeat,
        preserve_pitch=preserve_pitch,
        target_duration=requested_duration,
        output_duration=len(audio) / final_sr,
        method_sample_rate=sr,
        method_native_samples=method_native_samples,
        method_native_duration=method_native_duration,
        method_time_scale=method_time_scale,
        source_name=validated_source.name,
        postprocess=None if postprocess_spec is None else postprocess_spec.name,
        postprocess_params=(
            None
            if effective_postprocess_params is None
            else _freeze_parameters(effective_postprocess_params)
        ),
        postprocess_native_samples=postprocess_native_samples,
        postprocess_native_duration=postprocess_native_duration,
        postprocess_time_scale=postprocess_time_scale,
        output_path=output_path,
    )


__all__ = [
    "SonificationResult",
    "sonify",
]
