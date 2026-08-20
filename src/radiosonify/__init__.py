"""RadioSonify：把一维轮廓和多维数值数组转换为时长可控的音频。

推荐从 :func:`sonify` 和 :class:`SonificationInput` 开始；底层方法继续保留，
用于需要方法原生时长或复现旧结果的场景。
"""

from __future__ import annotations

import importlib

__version__ = "0.2.0"

from .amplitude import amplitude_modulate
from .api import SonificationResult, sonify
from .core import (
    del_burst,
    normalize,
    rebin_spectrogram,
    save_audio,
    to_profile,
)
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
    "profile_to_wave": (".profile", "profile_to_wave"),
    "erb_sonify": (".erb", "erb_sonify"),
    "erb_frequencies": (".erb", "erb_frequencies"),
    "mel_frequencies": (".erb", "mel_frequencies"),
    "griffinlim_reconstruct": (".griffinlim", "griffinlim"),
    "hifigan_vocode": (".hifigan", "hifigan"),
    "load_example": (".hub", "load_example"),
    "musicnet_transform": (".musicnet", "musicnet"),
    "rave_transform": (".rave", "rave"),
    "spatial_sonify": (".spatial", "spatial_sonify"),
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
    "del_burst",
    "rebin_spectrogram",
    "to_profile",
    "save_audio",
    "load_example",
    "profile_to_wave",
    "amplitude_modulate",
    "erb_sonify",
    "erb_frequencies",
    "mel_frequencies",
    "griffinlim_reconstruct",
    "hifigan_vocode",
    "musicnet_transform",
    "rave_transform",
    "spatial_sonify",
    "STYLE_NAMES",
]
