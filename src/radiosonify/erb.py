"""Public perceptual scan for two-dimensional matrices."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ._perceptual import (
    _condition_synthesized_audio,
    _settings_from_mapping,
    _synthesize_prepared,
    erb_frequencies,
    mel_frequencies,
)
from ._perceptual_config import PERCEPTUAL_DEFAULT_DURATION, PERCEPTUAL_DEFAULTS
from .core import _wav_output_path, save_audio
from .preprocessing import _as_normalized_array


def erb_sonify(
    data: np.ndarray,
    sr: int = PERCEPTUAL_DEFAULTS["sr"],
    duration: float = PERCEPTUAL_DEFAULT_DURATION,
    min_freq: float = PERCEPTUAL_DEFAULTS["min_freq"],
    max_freq: float = PERCEPTUAL_DEFAULTS["max_freq"],
    n_bands: int | None = PERCEPTUAL_DEFAULTS["n_bands"],
    value_scale: str = PERCEPTUAL_DEFAULTS["value_scale"],
    gamma: float = PERCEPTUAL_DEFAULTS["gamma"],
    frequency_order: str = PERCEPTUAL_DEFAULTS["frequency_order"],
    frequency_scale: str = PERCEPTUAL_DEFAULTS["frequency_scale"],
    timbre: str = PERCEPTUAL_DEFAULTS["timbre"],
    mapping_level_db: float = PERCEPTUAL_DEFAULTS["mapping_level_db"],
    ambient_level_db: float = PERCEPTUAL_DEFAULTS["ambient_level_db"],
    voice_params: Mapping[str, Any] | None = PERCEPTUAL_DEFAULTS["voice_params"],
    event_voice: str = PERCEPTUAL_DEFAULTS["event_voice"],
    event_params: Mapping[str, Any] | None = PERCEPTUAL_DEFAULTS["event_params"],
    attack_ms: float = PERCEPTUAL_DEFAULTS["attack_ms"],
    release_ms: float = PERCEPTUAL_DEFAULTS["release_ms"],
    loudness_compensation_db: float = PERCEPTUAL_DEFAULTS["loudness_compensation_db"],
    rms_limit_dbfs: float = PERCEPTUAL_DEFAULTS["rms_limit_dbfs"],
    peak_limit_dbfs: float = PERCEPTUAL_DEFAULTS["peak_limit_dbfs"],
    output: str | Path | None = None,
) -> tuple[np.ndarray, int]:
    """Map a normalized time-by-feature matrix to a perceptual additive scan.

    Time remains time and ordered feature position controls pitch.  Absolute
    brightness supplies a quiet ambient component, while continuous positive
    deviation from each band's temporal median supplies audible detail.
    ``n_bands=None`` selects approximately one simultaneous voice per auditory
    ERB.  ``timbre`` changes the carrier waveform, and optional event decoration
    adds sparse accents from the same salience map.  Advanced waveform and
    event controls are grouped in ``voice_params`` and ``event_params``.
    """
    arguments = locals()
    output_path = None if output is None else _wav_output_path(output)
    matrix = _as_normalized_array(data, name="data", ndim=2)
    settings = _settings_from_mapping(arguments)
    raw_audio = _synthesize_prepared(matrix, settings=settings)
    audio = _condition_synthesized_audio(
        raw_audio,
        sr=settings.sr,
        rms_limit_dbfs=settings.rms_limit_dbfs,
        peak_limit_dbfs=settings.peak_limit_dbfs,
    )

    if output_path is not None:
        save_audio(audio, settings.sr, output_path)
    return audio, settings.sr


__all__ = ["erb_frequencies", "erb_sonify", "mel_frequencies"]
