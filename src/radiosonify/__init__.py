"""RadioSonify：把射电轮廓和动态谱转换为时长可控的音频。

推荐从 :func:`sonify` 和 :class:`SonificationInput` 开始；底层方法继续保留，
用于需要方法原生时长或复现旧结果的场景。
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

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
from .registry import (
    MethodSpec,
    PostprocessorSpec,
    available_methods,
    available_postprocessors,
    default_method,
)
from .timing import (
    condition_audio_output,
    duration_to_samples,
    fit_audio_duration,
    target_audio_duration,
)

_LAZY_EXPORTS = {
    "profile_to_wave": (".profile", "profile_to_wave"),
    "griffinlim": (".griffinlim", "griffinlim"),
    "hifigan": (".hifigan", "hifigan"),
    "load_example": (".hub", "load_example"),
    "musicnet": (".musicnet", "musicnet"),
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


class _RadioSonifyModule(ModuleType):
    """保持与子模块同名的惰性函数仍表现为包级函数。"""

    def __getattribute__(self, name: str):
        lazy_exports = ModuleType.__getattribute__(self, "_LAZY_EXPORTS")
        namespace = ModuleType.__getattribute__(self, "__dict__")
        if name in lazy_exports and isinstance(namespace.get(name), ModuleType):
            namespace.pop(name)
            return __getattr__(name)
        return ModuleType.__getattribute__(self, name)


sys.modules[__name__].__class__ = _RadioSonifyModule


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
    "duration_to_samples",
    "fit_audio_duration",
    "condition_audio_output",
    "normalize",
    "del_burst",
    "rebin_spectrogram",
    "to_profile",
    "save_audio",
    "load_example",
    "profile_to_wave",
    "amplitude_modulate",
    "griffinlim",
    "hifigan",
    "musicnet",
    "STYLE_NAMES",
]
