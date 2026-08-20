"""Resolve one immutable, fully validated sonification execution plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from .audio_io import _wav_output_path
from .inputs import DataType, SonificationInput
from .preprocessing import resolve_preprocess_params
from .registry import MethodSpec, PostprocessorSpec, resolve_method, resolve_postprocessor
from .timing import duration_to_frames, duration_to_samples, target_audio_duration
from .validation import (
    _boolean,
    _freeze_mapping,
    _merge_settings,
    _positive_float,
    _positive_int,
)


@dataclass(frozen=True)
class ResolvedSonificationPlan:
    """All public choices and runtime contracts resolved before data processing."""

    source: SonificationInput
    method: MethodSpec
    method_params: Mapping[str, Any]
    preprocess_params: Mapping[str, Any]
    planned_input_shape: tuple[int, ...]
    postprocessor: PostprocessorSpec | None
    postprocess_params: Mapping[str, Any] | None
    postprocess_runtime: Mapping[str, Any]
    speed: float
    repeat: int
    preserve_pitch: bool
    requested_duration: float
    primary_duration: float
    output_sr: int | None
    output_path: Path | None


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
    return SonificationInput(np.asarray(source), duration=data_duration, data_type=data_type)


def _effective_method_params(
    spec: MethodSpec,
    supplied: Mapping[str, Any] | None,
) -> dict[str, Any]:
    params = _merge_settings(
        spec.defaults,
        supplied,
        field_name="method_params",
        unknown_label=f"unknown parameter(s) for method '{spec.name}'",
    )
    for group, group_defaults in (spec.grouped_defaults or {}).items():
        params[group] = _merge_settings(
            group_defaults,
            params[group],
            field_name=group,
            unknown_label=f"unknown {group} key(s)",
        )
    return spec.validate_params(params)


def _effective_postprocessor(
    name: str | None,
    supplied: Mapping[str, Any] | None,
) -> tuple[PostprocessorSpec | None, dict[str, Any] | None]:
    if name is None:
        if supplied is not None:
            raise ValueError("postprocess_params requires postprocess to be selected")
        return None, None
    spec = resolve_postprocessor(name)
    params = _merge_settings(
        spec.defaults,
        supplied,
        field_name="postprocess_params",
        unknown_label=f"unknown {spec.label} postprocess parameter(s)",
    )
    return spec, spec.validate_params(params)


def _resolve_feature_rebin(
    requested: int | str | None,
    spec: MethodSpec,
    method_params: Mapping[str, Any],
) -> int | None:
    if spec.feature_geometry is not None:
        default_bins, max_bins = spec.resolve_feature_geometry(method_params)
        default_bins = _positive_int(default_bins, name="registered default feature bins")
        max_bins = _positive_int(max_bins, name="registered maximum feature bins")
        if default_bins > max_bins:
            raise RuntimeError(f"method '{spec.name}' registers invalid feature geometry")
        resolved = _feature_count(requested, default=default_bins)
        if resolved > max_bins:
            raise ValueError(
                f"feature_rebin ({resolved}) cannot exceed {max_bins} for method "
                f"'{spec.name}' with the current method_params"
            )
        return resolved
    if spec.input_feature_bins is not None:
        required = _positive_int(spec.input_feature_bins, name="registered input_feature_bins")
        resolved = _feature_count(requested, default=required)
        if resolved != required:
            raise ValueError(
                f"method '{spec.name}' requires feature_rebin={required}, got {resolved}"
            )
        return resolved
    if requested == "auto":
        raise ValueError(
            f"feature_rebin='auto' requires feature geometry, which method '{spec.name}' "
            "does not register"
        )
    return _optional_rebin_count(requested, name="feature_rebin")


def _optional_rebin_count(requested: int | str | None, *, name: str) -> int | None:
    """Narrow a validated rebin setting without evaluating a runtime union type."""
    if requested is None:
        return None
    if not isinstance(requested, int):
        raise RuntimeError(f"resolved {name} must be an integer")
    return requested


def _feature_count(requested: int | str | None, *, default: int) -> int:
    """Resolve an optional feature target after preprocessing validation."""
    if requested in (None, "auto"):
        return default
    if not isinstance(requested, int):
        raise RuntimeError("resolved feature_rebin must be an integer")
    return requested


def _resolve_time_rebin(
    requested: int | str | None,
    spec: MethodSpec,
    method_params: Mapping[str, Any],
    *,
    source_time_bins: int,
    primary_duration: float,
    repeat: int,
) -> int | None:
    if requested is None:
        requested = spec.default_time_rebin
    if requested != "auto":
        return _optional_rebin_count(requested, name="time_rebin")
    if spec.frame_geometry is None:
        raise ValueError(
            f"time_rebin='auto' requires frame geometry, which method '{spec.name}' "
            "does not register"
        )
    sample_rate, hop_length = spec.resolve_frame_geometry(method_params)
    automatic_bins = (
        duration_to_frames(primary_duration / repeat, sample_rate, hop_length)
        + spec.repeat_frame_overlap
    )
    return automatic_bins if spec.allow_frame_upsampling else min(automatic_bins, source_time_bins)


def _apply_method_geometry(
    preprocess_params: Mapping[str, Any],
    spec: MethodSpec,
    method_params: Mapping[str, Any],
    *,
    source_shape: tuple[int, ...],
    primary_duration: float,
    repeat: int,
) -> dict[str, Any]:
    params = dict(preprocess_params)
    if len(source_shape) > 1:
        params["feature_rebin"] = _resolve_feature_rebin(
            params["feature_rebin"],
            spec,
            method_params,
        )
    time_axis = 0 if len(source_shape) < 3 else 1
    params["time_rebin"] = _resolve_time_rebin(
        params["time_rebin"],
        spec,
        method_params,
        source_time_bins=source_shape[time_axis],
        primary_duration=primary_duration,
        repeat=repeat,
    )
    return params


def _planned_preprocessed_shape(
    source_shape: tuple[int, ...],
    params: Mapping[str, Any],
    *,
    repeat: int,
    repeat_overlap: int,
) -> tuple[int, ...]:
    shape = list(source_shape)
    if len(shape) == 3 and params["layer_rebin"] is not None:
        if params["layer_rebin"] > shape[0]:
            raise ValueError(
                f"layer_rebin ({params['layer_rebin']}) cannot exceed input layer count "
                f"({shape[0]})"
            )
        shape[0] = params["layer_rebin"]
    time_axis = 0 if len(shape) < 3 else 1
    if params["time_rebin"] is not None:
        shape[time_axis] = params["time_rebin"]
    if len(shape) > 1 and params["feature_rebin"] is not None:
        shape[-1] = params["feature_rebin"]
    if repeat > 1 and repeat_overlap >= shape[time_axis]:
        raise ValueError("repeat overlap must be smaller than the planned time-axis length")
    shape[time_axis] += (shape[time_axis] - repeat_overlap) * (repeat - 1)
    return tuple(shape)


def resolve_sonification_plan(
    source: SonificationInput | np.ndarray,
    *,
    data_duration: float | None,
    data_type: DataType | str | None,
    method: str,
    speed: float,
    repeat: int | None,
    preserve_pitch: bool,
    output_sr: int | None,
    preprocess_params: Mapping[str, Any] | None,
    method_params: Mapping[str, Any] | None,
    postprocess: str | None,
    postprocess_params: Mapping[str, Any] | None,
    output: str | Path | None,
) -> ResolvedSonificationPlan:
    """Resolve every static error and optional runtime before touching scientific data."""
    resolved_source = _coerce_source(source, data_duration=data_duration, data_type=data_type)
    resolved_type = cast(DataType, resolved_source.data_type)
    resolved_pitch = _boolean(preserve_pitch, name="preserve_pitch")
    resolved_speed = _positive_float(speed, name="speed")
    resolved_output_sr = None if output_sr is None else _positive_int(output_sr, name="output_sr")
    output_path = None if output is None else _wav_output_path(output)

    method_spec = resolve_method(method, resolved_type)
    resolved_repeat = (
        method_spec.default_repeat if repeat is None else _positive_int(repeat, name="repeat")
    )
    postprocessor, effective_postprocess = _effective_postprocessor(
        postprocess,
        postprocess_params,
    )
    requested_duration = target_audio_duration(
        resolved_source.duration,
        resolved_speed,
        resolved_repeat,
    )
    primary_duration = (
        target_audio_duration(resolved_source.duration, 1.0, resolved_repeat)
        if postprocessor is not None and postprocessor.apply_speed_after
        else requested_duration
    )

    effective_method = _effective_method_params(method_spec, method_params)
    requested_preprocess = resolve_preprocess_params(resolved_type, preprocess_params)
    effective_preprocess = _apply_method_geometry(
        requested_preprocess,
        method_spec,
        effective_method,
        source_shape=tuple(resolved_source.data.shape),
        primary_duration=primary_duration,
        repeat=resolved_repeat,
    )
    planned_shape = _planned_preprocessed_shape(
        tuple(resolved_source.data.shape),
        effective_preprocess,
        repeat=resolved_repeat,
        repeat_overlap=method_spec.repeat_frame_overlap,
    )
    method_spec.validate_context(effective_method, planned_shape)

    if resolved_output_sr is not None:
        try:
            duration_to_samples(requested_duration, resolved_output_sr)
        except ValueError as exc:
            raise ValueError(f"output_sr cannot represent the requested duration: {exc}") from exc

    method_spec.preflight(effective_method)
    runtime: dict[str, Any] = {}
    if postprocessor is not None:
        if effective_postprocess is None:
            raise RuntimeError("postprocess parameters were not resolved")
        primary_sample_rate = _positive_int(
            method_spec.resolve_output_sample_rate(effective_method),
            name=f"method '{method_spec.name}' output sample rate",
        )
        runtime = postprocessor.prepare(
            effective_postprocess,
            input_channels=method_spec.output_channels,
            input_sample_rate=primary_sample_rate,
            input_samples=duration_to_samples(primary_duration, primary_sample_rate),
        )

    return ResolvedSonificationPlan(
        source=resolved_source,
        method=method_spec,
        method_params=_freeze_mapping(effective_method),
        preprocess_params=_freeze_mapping(effective_preprocess),
        planned_input_shape=planned_shape,
        postprocessor=postprocessor,
        postprocess_params=(
            None if effective_postprocess is None else _freeze_mapping(effective_postprocess)
        ),
        postprocess_runtime=_freeze_mapping(runtime),
        speed=resolved_speed,
        repeat=resolved_repeat,
        preserve_pitch=resolved_pitch,
        requested_duration=requested_duration,
        primary_duration=primary_duration,
        output_sr=resolved_output_sr,
        output_path=output_path,
    )


__all__ = ["ResolvedSonificationPlan", "resolve_sonification_plan"]
