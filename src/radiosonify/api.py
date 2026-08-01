"""按数据类型选择方法、控制时长并记录来源的统一声化流程。"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .core import _as_finite_array, _boolean, _positive_int, _wav_output_path, save_audio
from .inputs import DataType, SonificationInput
from .registry import MethodSpec, PostprocessorSpec, resolve_method, resolve_postprocessor
from .timing import (
    condition_audio_output,
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
    method: str
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


def _merge_parameters(
    defaults: Mapping[str, Any],
    supplied: Mapping[str, Any] | None,
    *,
    field_name: str,
    unknown_label: str,
) -> dict[str, Any]:
    """合并默认值，同时统一校验参数容器、键类型和未知键。"""
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, Mapping):
        raise ValueError(f"{field_name} must be a mapping or None")
    if any(not isinstance(key, str) for key in supplied):
        raise ValueError(f"{field_name} keys must be strings")

    unknown = sorted(set(supplied) - set(defaults))
    if unknown:
        joined = ", ".join(unknown)
        allowed = ", ".join(defaults)
        raise ValueError(f"{unknown_label}: {joined}; allowed: {allowed}")
    params = dict(defaults)
    params.update(supplied)
    return params


def _effective_params(
    spec: MethodSpec,
    supplied: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if spec.name == "griffinlim" and isinstance(supplied, Mapping) and "n_mels" in supplied:
        warnings.warn(
            "method_params['n_mels'] is deprecated; use 'freq_rebin' instead",
            DeprecationWarning,
            stacklevel=3,
        )
        if "freq_rebin" in supplied:
            raise ValueError(
                "n_mels and freq_rebin cannot both be supplied for method 'griffinlim'; "
                "use freq_rebin"
            )
        supplied = dict(supplied)
        supplied["freq_rebin"] = supplied.pop("n_mels")
    return _merge_parameters(
        spec.defaults,
        supplied,
        field_name="method_params",
        unknown_label=f"unknown parameter(s) for method '{spec.name}'",
    )


def _effective_postprocess(
    postprocess: str | None,
    supplied: Mapping[str, Any] | None,
) -> tuple[PostprocessorSpec | None, dict[str, Any] | None]:
    """解析后处理并在主要声化开始前完成全部参数校验。"""
    if postprocess is None:
        if supplied:
            raise ValueError("postprocess_params requires postprocess='musicnet'")
        return None, None
    spec = resolve_postprocessor(postprocess)

    params = _merge_parameters(
        spec.defaults,
        supplied,
        field_name="postprocess_params",
        unknown_label=f"unknown {spec.label} postprocess parameter(s)",
    )
    return spec, spec.validate_params(params)


# ---------- 方法执行 ----------


def _run_primary(
    source: SonificationInput,
    spec: MethodSpec,
    target_duration: float,
    repeat: int,
    params: dict[str, Any],
) -> tuple[np.ndarray, int]:
    runner = spec.load_runner()
    controls: dict[str, Any] = {"output": None}
    if spec.synthesizes_duration:
        controls.update(duration=target_duration, repeat=repeat)
    return runner(source.data, **controls, **params)


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


# ---------- 公开统一入口 ----------


def sonify(
    source: SonificationInput | np.ndarray,
    *,
    data_duration: float | None = None,
    data_type: DataType | str | None = None,
    method: str = "auto",
    speed: float = 1.0,
    repeat: int = 1,
    preserve_pitch: bool = False,
    output_sr: int | None = None,
    method_params: Mapping[str, Any] | None = None,
    postprocess: str | None = None,
    postprocess_params: Mapping[str, Any] | None = None,
    output: str | Path | None = None,
) -> SonificationResult:
    """通过统一流程声化一个轮廓或动态谱。

    Arrays require ``data_duration``. One-dimensional arrays are inferred as
    profiles and two-dimensional arrays as dynamic spectra unless ``data_type``
    is explicitly supplied. The final sample count is always
    ``round(sample_rate * data_duration * repeat / speed)``. ``repeat`` is
    supported by the transparent profile and amplitude methods. ``output_sr``
    optionally converts the final container sample rate without synthesizing
    bandwidth above the method's native Nyquist frequency.
    """
    validated_source = _coerce_source(
        source,
        data_duration=data_duration,
        data_type=data_type,
    )
    preserve_pitch = _boolean(preserve_pitch, name="preserve_pitch")
    repeat = _positive_int(repeat, name="repeat")
    if output_sr is not None:
        output_sr = _positive_int(output_sr, name="output_sr")
    # 输出路径也必须在耗时合成前失败，避免最后一步才发现扩展名或目录错误。
    output_path = None if output is None else _wav_output_path(output)

    spec = resolve_method(method, validated_source.data_type)
    if repeat != 1 and not spec.supports_repeat:
        raise ValueError("repeat is only supported by the profile and amplitude methods")
    params = _effective_params(spec, method_params)
    postprocess_spec, effective_postprocess_params = _effective_postprocess(
        postprocess,
        postprocess_params,
    )
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
    audio, sr = _run_primary(
        validated_source,
        spec,
        primary_duration,
        repeat,
        params,
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
            _as_finite_array(audio, name="postprocessed audio", ndim=1),
            final_sr,
            requested_duration,
            preserve_pitch=preserve_pitch,
        )
        postprocess_time_scale = len(audio) / postprocess_native_samples

    if output_sr is not None and output_sr != final_sr:
        # 这里只改变容器采样率，不改变物理时长或音高。相位声码器按旧采样率
        # 拉长后再用新采样率播放会把音高抬高 output_sr / final_sr 倍，
        # 因此采样率转换必须固定使用多相路径。
        audio = fit_audio_duration(
            audio,
            output_sr,
            requested_duration,
            preserve_pitch=False,
        )
        final_sr = output_sr
    audio = condition_audio_output(audio, final_sr)

    if output_path is not None:
        save_audio(audio, final_sr, output_path)
    # 与 SonificationInput 一致，冻结数组内容才能让结果的溯源字段保持可信。
    audio.setflags(write=False)

    return SonificationResult(
        audio=audio,
        sample_rate=final_sr,
        data_type=validated_source.data_type,
        data_duration=validated_source.duration,
        method=spec.name,
        method_params=MappingProxyType(dict(params)),
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
            else MappingProxyType(dict(effective_postprocess_params))
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
