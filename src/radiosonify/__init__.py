"""RadioSonify: duration-aware audio mapping for 1-D, 2-D, and 3-D arrays."""

from __future__ import annotations

import importlib

__version__ = "0.3.0"

from .api import SonificationResult, sonify
from .array_ops import normalize, to_profile
from .audio_io import save_audio
from .inputs import DataType, SonificationInput, infer_data_type
from .preprocessing import preprocess, preprocessing_defaults
from .registry import (
    MethodSpec,
    PostprocessorSpec,
    available_methods,
    available_postprocessors,
    default_method,
)
from .timing import (
    condition_audio_output,
    duration_to_frames,
    duration_to_samples,
    fit_audio_duration,
    target_audio_duration,
)

_LAZY_EXPORTS = {
    "erb_frequencies": (".erb", "erb_frequencies"),
    "mel_frequencies": (".erb", "mel_frequencies"),
    "load_example": (".hub", "load_example"),
    "STYLE_NAMES": (".musicnet", "STYLE_NAMES"),
}


def __getattr__(name: str):
    """仅在首次访问重型/联网功能时导入对应模块。"""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "__version__",
    "DataType",
    "SonificationInput",
    "SonificationResult",
    "MethodSpec",
    "PostprocessorSpec",
    "sonify",
    "available_methods",
    "available_postprocessors",
    "default_method",
    "infer_data_type",
    "target_audio_duration",
    "duration_to_frames",
    "duration_to_samples",
    "fit_audio_duration",
    "condition_audio_output",
    "preprocess",
    "preprocessing_defaults",
    "normalize",
    "to_profile",
    "save_audio",
    "load_example",
    "erb_frequencies",
    "mel_frequencies",
    "STYLE_NAMES",
]
