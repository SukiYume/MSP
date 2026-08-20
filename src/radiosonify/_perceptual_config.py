"""Shared immutable defaults and choices for perceptual sonification.

The public mapping keeps coordinate, contrast, and output controls together.
Voice synthesis details and optional event decoration live in their own
mappings.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

PERCEPTUAL_CHOICES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "value_scale": ("amplitude", "power"),
        "frequency_order": ("ascending", "descending"),
        "frequency_scale": ("mel", "erb"),
        "timbre": (
            "retro_digital",
            "sine",
            "warm_pad",
            "soft_marimba",
            "glass_bell",
            "instrument_palette",
        ),
        "event_voice": ("none", "water_drop"),
    }
)

PERCEPTUAL_DEFAULT_DURATION = 2.0

PERCEPTUAL_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "sr": 48_000,
        "min_freq": 100.0,
        "max_freq": 2_000.0,
        "n_bands": None,
        "value_scale": "amplitude",
        "gamma": 4.0,
        "frequency_order": "ascending",
        "frequency_scale": "mel",
        "timbre": "sine",
        "mapping_level_db": 0.0,
        "ambient_level_db": -30.0,
        "voice_params": None,
        "event_voice": "none",
        "event_params": None,
        "attack_ms": 8.0,
        "release_ms": 80.0,
        "loudness_compensation_db": 6.0,
        "rms_limit_dbfs": -20.0,
        "peak_limit_dbfs": -1.0,
    }
)

VOICE_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "harmonic_limit_hz": 3_500.0,
        "detune_cents": 10.0,
        "fm_index": 1.0,
        "chorus_rate_hz": 0.45,
        "chorus_depth_ms": 8.0,
    }
)

EVENT_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "salience_threshold": 0.35,
        "max_events_per_second": 6.0,
        "decay_ms": 70.0,
        "level_db": -20.0,
    }
)

__all__ = [
    "PERCEPTUAL_CHOICES",
    "PERCEPTUAL_DEFAULT_DURATION",
    "PERCEPTUAL_DEFAULTS",
    "VOICE_DEFAULTS",
    "EVENT_DEFAULTS",
]
