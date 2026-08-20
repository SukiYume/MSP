"""Deterministic, band-limited foreground voices for perceptual sonification."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .validation import _finite_float, _positive_int

_GOLDEN_RATIO_CONJUGATE = (np.sqrt(5.0) - 1.0) / 2.0
_MAX_RETRO_HARMONICS = 6
_RETRO_CORE_GAIN = 0.78
_RETRO_DETUNE_GAIN = 0.11
_DETUNE_PHASE_OFFSETS = (0.37, -0.61)
_PALETTE_MIDPOINT = 0.5


def _phase_modulation_coefficients(
    modulation_index: float,
    n_harmonics: int,
) -> np.ndarray:
    """Expand sinusoidal phase modulation into a normalized harmonic series."""
    from scipy.special import jv

    resolved_index = _finite_float(modulation_index, name="modulation_index")
    if not 0.0 <= resolved_index <= 1.0:
        raise ValueError("modulation_index must be between 0 and 1")
    resolved_harmonics = _positive_int(n_harmonics, name="n_harmonics")
    harmonic_numbers = np.arange(1, resolved_harmonics + 1)
    coefficients = jv(harmonic_numbers - 1, resolved_index) + ((-1.0) ** harmonic_numbers) * jv(
        harmonic_numbers + 1, resolved_index
    )
    norm = float(np.linalg.norm(coefficients))
    if norm <= np.finfo(np.float64).eps:
        coefficients = np.zeros(resolved_harmonics, dtype=np.float64)
        coefficients[0] = 1.0
        return coefficients
    return np.asarray(coefficients / norm, dtype=np.float64)


def _render_additive_voice(
    sample_time: np.ndarray,
    *,
    frequency: float,
    phase: float,
    harmonic_limit_hz: float,
    ratios: Sequence[float],
    gains: Sequence[float],
    phase_offsets: Sequence[float],
) -> np.ndarray:
    """Render normalized sinusoidal modes while omitting out-of-band overtones."""
    if not len(ratios) == len(gains) == len(phase_offsets):
        raise RuntimeError("additive voice ratios, gains, and phase offsets must align")
    voice = np.zeros_like(sample_time, dtype=np.float64)
    squared_gain = 0.0
    for index, (ratio, gain, offset) in enumerate(zip(ratios, gains, phase_offsets)):
        partial_frequency = frequency * ratio
        # The fundamental always represents the requested pitch. Higher modes
        # obey the shared harmonic ceiling so a voice cannot add aliased energy.
        if index > 0 and partial_frequency > harmonic_limit_hz:
            continue
        voice += gain * np.sin(2.0 * np.pi * partial_frequency * sample_time + phase + offset)
        squared_gain += gain**2
    if squared_gain == 0:  # Defensive: every built-in voice includes a fundamental.
        return voice
    return voice / np.sqrt(squared_gain)


def _render_retro_voice(
    sample_time: np.ndarray,
    *,
    frequency: float,
    phase: float,
    band_index: int,
    harmonic_limit_hz: float,
    detune_cents: float,
    fm_index: float,
    chorus_rate_hz: float,
    chorus_depth_ms: float,
) -> np.ndarray:
    """Render one deterministic band-limited retro digital foreground voice."""
    max_harmonic = max(
        1,
        min(
            _MAX_RETRO_HARMONICS,
            int(np.floor(harmonic_limit_hz / frequency)),
        ),
    )
    coefficients = _phase_modulation_coefficients(fm_index, max_harmonic)
    base_phase = 2.0 * np.pi * frequency * sample_time + phase
    core = np.zeros_like(sample_time, dtype=np.float64)
    for harmonic, coefficient in enumerate(coefficients, start=1):
        core += coefficient * np.sin(harmonic * base_phase)

    voice = _RETRO_CORE_GAIN * core
    total_gain = _RETRO_CORE_GAIN
    lfo_phase = 2.0 * np.pi * (((band_index + 0.5) * _GOLDEN_RATIO_CONJUGATE) % 1.0)
    chorus_delay = (chorus_depth_ms / 1_000.0) * np.sin(
        2.0 * np.pi * chorus_rate_hz * sample_time + lfo_phase
    )
    detune_ratios = (
        2.0 ** (-detune_cents / 1_200.0),
        2.0 ** (detune_cents / 1_200.0),
    )
    for direction, ratio, phase_offset in zip(
        (-1.0, 1.0),
        detune_ratios,
        _DETUNE_PHASE_OFFSETS,
    ):
        detuned_frequency = frequency * ratio
        if detuned_frequency > harmonic_limit_hz:
            continue
        detuned_phase = (
            2.0 * np.pi * detuned_frequency * (sample_time + direction * chorus_delay)
            + phase
            + phase_offset
        )
        voice += _RETRO_DETUNE_GAIN * np.sin(detuned_phase)
        total_gain += _RETRO_DETUNE_GAIN
    return voice / total_gain


def _render_warm_pad(
    sample_time: np.ndarray,
    *,
    frequency: float,
    phase: float,
    band_index: int,
    harmonic_limit_hz: float,
    detune_cents: float,
    chorus_rate_hz: float,
    chorus_depth_ms: float,
) -> np.ndarray:
    """Render a warm sustained voice with restrained harmonics and chorus."""
    core_gains = (1.0, 0.20, 0.09, 0.04)
    core = _render_additive_voice(
        sample_time,
        frequency=frequency,
        phase=phase,
        harmonic_limit_hz=harmonic_limit_hz,
        ratios=(1.0, 2.0, 3.0, 4.0),
        gains=core_gains,
        phase_offsets=(0.0, 0.18, -0.31, 0.47),
    )
    if detune_cents == 0:
        return core

    lfo_phase = 2.0 * np.pi * (((band_index + 0.25) * _GOLDEN_RATIO_CONJUGATE) % 1.0)
    delay = (chorus_depth_ms / 2_000.0) * np.sin(
        2.0 * np.pi * chorus_rate_hz * sample_time + lfo_phase
    )
    side_gain = 0.10
    lower = frequency * 2.0 ** (-0.55 * detune_cents / 1_200.0)
    upper = frequency * 2.0 ** (0.55 * detune_cents / 1_200.0)
    side = np.zeros_like(sample_time, dtype=np.float64)
    side_count = 0
    if lower <= harmonic_limit_hz:
        side += np.sin(2.0 * np.pi * lower * (sample_time - delay) + phase + 0.29)
        side_count += 1
    if upper <= harmonic_limit_hz:
        side += np.sin(2.0 * np.pi * upper * (sample_time + delay) + phase - 0.41)
        side_count += 1
    # The core and two side oscillators are normalized by coefficient energy,
    # keeping the same envelope meaningful when users switch voices.
    return (core + side_gain * side) / np.sqrt(1.0 + side_count * side_gain**2)


def _render_soft_marimba(
    sample_time: np.ndarray,
    *,
    frequency: float,
    phase: float,
    harmonic_limit_hz: float,
) -> np.ndarray:
    """Render a soft marimba-like set of tuned and inharmonic modes."""
    return _render_additive_voice(
        sample_time,
        frequency=frequency,
        phase=phase,
        harmonic_limit_hz=harmonic_limit_hz,
        ratios=(1.0, 3.92, 9.17),
        gains=(1.0, 0.30, 0.10),
        phase_offsets=(0.0, 1.08, -0.73),
    )


def _render_glass_bell(
    sample_time: np.ndarray,
    *,
    frequency: float,
    phase: float,
    harmonic_limit_hz: float,
) -> np.ndarray:
    """Render a clear glass-bell-like inharmonic modal spectrum."""
    return _render_additive_voice(
        sample_time,
        frequency=frequency,
        phase=phase,
        harmonic_limit_hz=harmonic_limit_hz,
        ratios=(1.0, 2.01, 2.72, 4.08, 5.43),
        gains=(0.78, 0.48, 0.30, 0.16, 0.09),
        phase_offsets=(0.0, 0.63, -0.44, 1.17, -1.02),
    )


def _palette_components(position: float) -> tuple[tuple[str, float], ...]:
    """Return a continuous low-pad/mid-marimba/high-glass crossfade."""
    resolved = _finite_float(position, name="position")
    if not 0.0 <= resolved <= 1.0:
        raise ValueError("position must be between 0 and 1")
    if resolved <= _PALETTE_MIDPOINT:
        blend = resolved / _PALETTE_MIDPOINT
        components = (("warm_pad", 1.0 - blend), ("soft_marimba", blend))
    else:
        blend = (resolved - _PALETTE_MIDPOINT) / (1.0 - _PALETTE_MIDPOINT)
        components = (("soft_marimba", 1.0 - blend), ("glass_bell", blend))
    return tuple((name, weight) for name, weight in components if weight > 0.0)


def _render_voice(
    name: str,
    sample_time: np.ndarray,
    *,
    frequency: float,
    phase: float,
    band_index: int,
    band_position: float,
    harmonic_limit_hz: float,
    detune_cents: float,
    fm_index: float,
    chorus_rate_hz: float,
    chorus_depth_ms: float,
) -> np.ndarray:
    """Render a named voice or the continuous instrument palette."""
    if name == "sine":
        return np.sin(2.0 * np.pi * frequency * sample_time + phase)
    if name == "retro_digital":
        return _render_retro_voice(
            sample_time,
            frequency=frequency,
            phase=phase,
            band_index=band_index,
            harmonic_limit_hz=harmonic_limit_hz,
            detune_cents=detune_cents,
            fm_index=fm_index,
            chorus_rate_hz=chorus_rate_hz,
            chorus_depth_ms=chorus_depth_ms,
        )
    if name == "warm_pad":
        return _render_warm_pad(
            sample_time,
            frequency=frequency,
            phase=phase,
            band_index=band_index,
            harmonic_limit_hz=harmonic_limit_hz,
            detune_cents=detune_cents,
            chorus_rate_hz=chorus_rate_hz,
            chorus_depth_ms=chorus_depth_ms,
        )
    if name == "soft_marimba":
        return _render_soft_marimba(
            sample_time,
            frequency=frequency,
            phase=phase,
            harmonic_limit_hz=harmonic_limit_hz,
        )
    if name == "glass_bell":
        return _render_glass_bell(
            sample_time,
            frequency=frequency,
            phase=phase,
            harmonic_limit_hz=harmonic_limit_hz,
        )
    if name == "instrument_palette":
        voice = np.zeros_like(sample_time, dtype=np.float64)
        for component, weight in _palette_components(band_position):
            voice += weight * _render_voice(
                component,
                sample_time,
                frequency=frequency,
                phase=phase,
                band_index=band_index,
                band_position=band_position,
                harmonic_limit_hz=harmonic_limit_hz,
                detune_cents=detune_cents,
                fm_index=fm_index,
                chorus_rate_hz=chorus_rate_hz,
                chorus_depth_ms=chorus_depth_ms,
            )
        return voice
    raise ValueError(f"unknown voice: {name}")


__all__ = [
    "_GOLDEN_RATIO_CONJUGATE",
    "_palette_components",
    "_phase_modulation_coefficients",
    "_render_retro_voice",
    "_render_voice",
]
