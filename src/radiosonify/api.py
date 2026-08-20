"""Public duration-aware sonification API and immutable result provenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from .inputs import DataType, SonificationInput
from .pipeline import execute_sonification_plan
from .planning import resolve_sonification_plan
from .validation import _freeze_mapping


@dataclass(frozen=True, eq=False)
class SonificationResult:
    """Final audio and all effective settings required to reproduce it."""

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


def _freeze_parameters(params: Mapping[str, Any]) -> Mapping[str, Any]:
    return _freeze_mapping(params)


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
    """Convert a 1-D profile, 2-D matrix, or 3-D layer stack to audio.

    The call first resolves an immutable execution plan. Parameter ranges,
    optional runtimes, method geometry, model channel contracts, output paths,
    and the final planned preprocessing shape are checked before the scientific
    array is copied or transformed. Execution then follows the fixed sequence
    preprocessing, primary synthesis, duration fitting, optional aesthetic
    postprocessing, sample-rate conversion, and output conditioning.
    """
    plan = resolve_sonification_plan(
        source,
        data_duration=data_duration,
        data_type=data_type,
        method=method,
        speed=speed,
        repeat=repeat,
        preserve_pitch=preserve_pitch,
        output_sr=output_sr,
        preprocess_params=preprocess_params,
        method_params=method_params,
        postprocess=postprocess,
        postprocess_params=postprocess_params,
        output=output,
    )
    execution = execute_sonification_plan(plan)
    resolved_type = cast(DataType, plan.source.data_type)
    method_params_with_provenance = {
        **dict(plan.method_params),
        **execution.method_provenance,
    }

    return SonificationResult(
        audio=execution.audio,
        sample_rate=execution.sample_rate,
        data_type=resolved_type,
        data_duration=plan.source.duration,
        input_shape=plan.source.input_shape,
        source_time_axis=plan.source.source_time_axis,
        source_layer_axis=plan.source.source_layer_axis,
        method=plan.method.name,
        preprocess_params=_freeze_parameters(plan.preprocess_params),
        method_params=_freeze_parameters(method_params_with_provenance),
        speed=plan.speed,
        repeat=plan.repeat,
        preserve_pitch=plan.preserve_pitch,
        target_duration=plan.requested_duration,
        output_duration=len(execution.audio) / execution.sample_rate,
        method_sample_rate=execution.method_sample_rate,
        method_native_samples=execution.method_native_samples,
        method_native_duration=execution.method_native_duration,
        method_time_scale=execution.method_time_scale,
        source_name=plan.source.name,
        postprocess=None if plan.postprocessor is None else plan.postprocessor.name,
        postprocess_params=(
            None if plan.postprocess_params is None else _freeze_parameters(plan.postprocess_params)
        ),
        postprocess_native_samples=execution.postprocess_native_samples,
        postprocess_native_duration=execution.postprocess_native_duration,
        postprocess_time_scale=execution.postprocess_time_scale,
        output_path=plan.output_path,
    )


__all__ = ["SonificationResult", "sonify"]
