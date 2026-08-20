"""Execute a resolved sonification plan without resolving public policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .audio_io import save_audio
from .planning import ResolvedSonificationPlan
from .preprocessing import _preprocess_validated
from .timing import (
    _resample_audio_rate,
    condition_audio_output,
    duration_to_samples,
    fit_audio_duration,
)
from .validation import _as_finite_array, _immutable_array, _positive_int


@dataclass(frozen=True)
class PipelineExecution:
    audio: np.ndarray
    sample_rate: int
    method_provenance: dict[str, Any]
    method_sample_rate: int
    method_native_samples: int
    method_native_duration: float
    method_time_scale: float
    postprocess_native_samples: int | None
    postprocess_native_duration: float | None
    postprocess_time_scale: float | None


def _run_primary(
    plan: ResolvedSonificationPlan,
    data: np.ndarray,
    provenance: dict[str, Any],
) -> tuple[np.ndarray, int]:
    controls: dict[str, Any] = {"output": None}
    if plan.method.synthesizes_duration:
        controls["duration"] = plan.primary_duration
    if plan.method.emits_provenance:
        controls["provenance"] = provenance
    return plan.method.load_runner()(data, **controls, **dict(plan.method_params))


def _run_postprocessor(
    plan: ResolvedSonificationPlan,
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, int, int, float, float]:
    postprocessor = plan.postprocessor
    params = plan.postprocess_params
    if postprocessor is None or params is None:
        raise RuntimeError("postprocessor execution requires a resolved postprocessor plan")
    processed, output_sr = postprocessor.load_runner()(
        audio,
        sr=sample_rate,
        output=None,
        **dict(params),
        **dict(plan.postprocess_runtime),
    )
    output_sr = _positive_int(output_sr, name="postprocess sample rate")
    finite = _as_finite_array(processed, name="postprocessed audio", ndim=(1, 2))
    native_samples = len(finite)
    fitted = fit_audio_duration(
        finite,
        output_sr,
        plan.requested_duration,
        preserve_pitch=plan.preserve_pitch,
    )
    return (
        fitted,
        output_sr,
        native_samples,
        native_samples / output_sr,
        len(fitted) / native_samples,
    )


def execute_sonification_plan(plan: ResolvedSonificationPlan) -> PipelineExecution:
    """Run preprocessing, synthesis, timing, postprocessing, and output conditioning."""
    data = _preprocess_validated(
        plan.source.data,
        plan.preprocess_params,
        repeat=plan.repeat,
        repeat_overlap=plan.method.repeat_frame_overlap,
    )
    if data.shape != plan.planned_input_shape:
        raise RuntimeError(
            f"preprocessing produced shape {data.shape}, expected {plan.planned_input_shape}"
        )
    data.setflags(write=False)

    method_provenance: dict[str, Any] = {}
    audio, method_sr = _run_primary(plan, data, method_provenance)
    method_sr = _positive_int(method_sr, name="method sample rate")
    audio = _as_finite_array(audio, name="method audio", ndim=(1, 2))
    method_channels = 1 if audio.ndim == 1 else audio.shape[1]
    if method_channels != plan.method.output_channels:
        raise RuntimeError(
            f"method '{plan.method.name}' produced {method_channels} channel(s); "
            f"its registered contract declares {plan.method.output_channels}"
        )
    method_native_samples = len(audio)
    method_native_duration = method_native_samples / method_sr
    audio = fit_audio_duration(
        audio,
        method_sr,
        plan.primary_duration,
        preserve_pitch=plan.preserve_pitch,
    )
    method_time_scale = len(audio) / method_native_samples

    final_sr = method_sr
    postprocess_native_samples = None
    postprocess_native_duration = None
    postprocess_time_scale = None
    if plan.postprocessor is not None:
        (
            audio,
            final_sr,
            postprocess_native_samples,
            postprocess_native_duration,
            postprocess_time_scale,
        ) = _run_postprocessor(plan, audio, method_sr)

    if plan.output_sr is not None and plan.output_sr != final_sr:
        audio = _resample_audio_rate(
            audio,
            final_sr,
            plan.output_sr,
            target_samples=duration_to_samples(plan.requested_duration, plan.output_sr),
        )
        final_sr = plan.output_sr

    output_peak = (
        plan.method.output_peak if plan.postprocessor is None else plan.postprocessor.output_peak
    )
    audio = condition_audio_output(audio, final_sr, peak=output_peak)
    if plan.output_path is not None:
        save_audio(audio, final_sr, plan.output_path)

    return PipelineExecution(
        audio=_immutable_array(audio),
        sample_rate=final_sr,
        method_provenance=method_provenance,
        method_sample_rate=method_sr,
        method_native_samples=method_native_samples,
        method_native_duration=method_native_duration,
        method_time_scale=method_time_scale,
        postprocess_native_samples=postprocess_native_samples,
        postprocess_native_duration=postprocess_native_duration,
        postprocess_time_scale=postprocess_time_scale,
    )


__all__ = ["PipelineExecution", "execute_sonification_plan"]
