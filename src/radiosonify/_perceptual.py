"""Shared perceptual coordinate mapping and additive synthesis engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._events import _add_event_layer
from ._perceptual_config import (
    EVENT_DEFAULTS,
    PERCEPTUAL_CHOICES,
    PERCEPTUAL_DEFAULTS,
    VOICE_DEFAULTS,
)
from ._voices import _render_voice
from .core import (
    _MAD_TO_GAUSSIAN_SIGMA,
    _choice,
    _finite_float,
    _merge_settings,
    _positive_float,
    _positive_int,
    _rebin_axis,
)
from .timing import condition_audio_output, duration_to_samples

_MIN_ROBUST_SCALE = 1e-6
_SALIENCE_SATURATION_SIGMA = 2.0
_TRUE_PEAK_OVERSAMPLE = 4

_VALUE_SCALES = set(PERCEPTUAL_CHOICES["value_scale"])
_FREQUENCY_ORDERS = set(PERCEPTUAL_CHOICES["frequency_order"])
_FREQUENCY_SCALES = set(PERCEPTUAL_CHOICES["frequency_scale"])
_TIMBRES = set(PERCEPTUAL_CHOICES["timbre"])
_EVENT_VOICES = set(PERCEPTUAL_CHOICES["event_voice"])


@dataclass(frozen=True)
class _VoiceSettings:
    """Procedural waveform controls, independent of data selection."""

    name: str
    harmonic_limit_hz: float
    detune_cents: float
    fm_index: float
    chorus_rate_hz: float
    chorus_depth_ms: float


@dataclass(frozen=True)
class _EventSettings:
    """Optional decoration controls, independent of the continuous mapping."""

    voice: str
    salience_threshold: float
    max_events_per_second: float
    decay_ms: float
    level_db: float


@dataclass(frozen=True)
class _SynthesisSettings:
    """Validated settings shared by mono and spatial synthesis."""

    sr: int
    duration: float
    min_freq: float
    max_freq: float
    n_bands: int
    value_scale: str
    gamma: float
    frequency_order: str
    frequency_scale: str
    mapping_level_db: float
    ambient_level_db: float
    attack_ms: float
    release_ms: float
    loudness_compensation_db: float
    rms_limit_dbfs: float
    peak_limit_dbfs: float
    voice: _VoiceSettings
    event: _EventSettings


def _nonnegative_float(value: float, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result < 0:
        raise ValueError(f"{name} must be a finite number greater than or equal to 0")
    return result


def _nonpositive_db(value: float, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result > 0:
        raise ValueError(f"{name} must be less than or equal to 0 dB")
    return result


def _bounded_float(
    value: float,
    *,
    name: str,
    lower: float,
    upper: float,
) -> float:
    result = _finite_float(value, name=name)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be between {lower:g} and {upper:g}")
    return result


def _resolve_voice_settings(
    timbre: str,
    voice_params: Mapping[str, Any] | None,
) -> _VoiceSettings:
    params = _merge_settings(
        VOICE_DEFAULTS,
        voice_params,
        field_name="voice_params",
        unknown_label="unknown voice_params key(s)",
    )
    return _VoiceSettings(
        name=_choice(timbre, name="timbre", choices=_TIMBRES),
        harmonic_limit_hz=_positive_float(
            params["harmonic_limit_hz"],
            name="voice_params.harmonic_limit_hz",
        ),
        detune_cents=_bounded_float(
            params["detune_cents"],
            name="voice_params.detune_cents",
            lower=0.0,
            upper=50.0,
        ),
        fm_index=_bounded_float(
            params["fm_index"],
            name="voice_params.fm_index",
            lower=0.0,
            upper=1.0,
        ),
        chorus_rate_hz=_bounded_float(
            params["chorus_rate_hz"],
            name="voice_params.chorus_rate_hz",
            lower=0.0,
            upper=10.0,
        ),
        chorus_depth_ms=_bounded_float(
            params["chorus_depth_ms"],
            name="voice_params.chorus_depth_ms",
            lower=0.0,
            upper=20.0,
        ),
    )


def _resolve_event_settings(
    event_voice: str,
    event_params: Mapping[str, Any] | None,
) -> _EventSettings:
    params = _merge_settings(
        EVENT_DEFAULTS,
        event_params,
        field_name="event_params",
        unknown_label="unknown event_params key(s)",
    )
    return _EventSettings(
        voice=_choice(event_voice, name="event_voice", choices=_EVENT_VOICES),
        salience_threshold=_bounded_float(
            params["salience_threshold"],
            name="event_params.salience_threshold",
            lower=0.0,
            upper=1.0,
        ),
        max_events_per_second=_bounded_float(
            params["max_events_per_second"],
            name="event_params.max_events_per_second",
            lower=0.0,
            upper=100.0,
        ),
        decay_ms=_bounded_float(
            params["decay_ms"],
            name="event_params.decay_ms",
            lower=1.0,
            upper=5_000.0,
        ),
        level_db=_nonpositive_db(
            params["level_db"],
            name="event_params.level_db",
        ),
    )


def _resolve_synthesis_settings(
    *,
    sr: int,
    duration: float,
    min_freq: float,
    max_freq: float,
    n_bands: int | None,
    value_scale: str,
    gamma: float,
    frequency_order: str,
    frequency_scale: str,
    timbre: str,
    mapping_level_db: float,
    ambient_level_db: float,
    voice_params: Mapping[str, Any] | None,
    event_voice: str,
    event_params: Mapping[str, Any] | None,
    attack_ms: float,
    release_ms: float,
    loudness_compensation_db: float,
    rms_limit_dbfs: float,
    peak_limit_dbfs: float,
) -> _SynthesisSettings:
    """Validate the compact mapping controls and two optional parameter groups."""
    resolved_sr = _positive_int(sr, name="sr")
    resolved_duration = _positive_float(duration, name="duration")
    resolved_min = _positive_float(min_freq, name="min_freq")
    resolved_max = _positive_float(max_freq, name="max_freq")
    if resolved_max >= resolved_sr / 2:
        raise ValueError(f"max_freq must be below the Nyquist frequency ({resolved_sr / 2:g} Hz)")
    if resolved_max <= resolved_min:
        raise ValueError("max_freq must be greater than min_freq")

    resolved_n_bands = (
        _auditory_band_count(resolved_min, resolved_max)
        if n_bands is None
        else _positive_int(n_bands, name="n_bands")
    )

    return _SynthesisSettings(
        sr=resolved_sr,
        duration=resolved_duration,
        min_freq=resolved_min,
        max_freq=resolved_max,
        n_bands=resolved_n_bands,
        value_scale=_choice(value_scale, name="value_scale", choices=_VALUE_SCALES),
        gamma=_positive_float(gamma, name="gamma"),
        frequency_order=_choice(
            frequency_order,
            name="frequency_order",
            choices=_FREQUENCY_ORDERS,
        ),
        frequency_scale=_choice(
            frequency_scale,
            name="frequency_scale",
            choices=_FREQUENCY_SCALES,
        ),
        mapping_level_db=_nonpositive_db(mapping_level_db, name="mapping_level_db"),
        ambient_level_db=_nonpositive_db(
            ambient_level_db,
            name="ambient_level_db",
        ),
        attack_ms=_nonnegative_float(attack_ms, name="attack_ms"),
        release_ms=_nonnegative_float(release_ms, name="release_ms"),
        loudness_compensation_db=_nonnegative_float(
            loudness_compensation_db,
            name="loudness_compensation_db",
        ),
        rms_limit_dbfs=_nonpositive_db(rms_limit_dbfs, name="rms_limit_dbfs"),
        peak_limit_dbfs=_nonpositive_db(peak_limit_dbfs, name="peak_limit_dbfs"),
        voice=_resolve_voice_settings(timbre, voice_params),
        event=_resolve_event_settings(event_voice, event_params),
    )


def _settings_from_mapping(values: Mapping[str, Any]) -> _SynthesisSettings:
    """Resolve the shared parameter subset from a public method's arguments."""
    required = ("duration", *PERCEPTUAL_DEFAULTS)
    missing = tuple(name for name in required if name not in values)
    if missing:
        raise RuntimeError(f"perceptual method omitted shared parameter(s): {', '.join(missing)}")
    return _resolve_synthesis_settings(
        duration=values["duration"],
        **{name: values[name] for name in PERCEPTUAL_DEFAULTS},
    )


def _prepare_values(
    data: np.ndarray,
    *,
    value_scale: str,
    gamma: float,
) -> np.ndarray:
    """Map an already validated normalized array onto its brightness curve."""
    exponent = gamma * (0.5 if value_scale == "power" else 1.0)
    return data**exponent


def _mapping_envelopes(
    data: np.ndarray,
    *,
    salience: np.ndarray,
    value_scale: str,
    gamma: float,
    level_db: float,
    ambient_level_db: float,
) -> np.ndarray:
    """Combine validated brightness and salience arrays into data envelopes."""
    gain = 10.0 ** (level_db / 20.0)
    ambient_gain = 10.0 ** (ambient_level_db / 20.0)
    brightness = _prepare_values(
        data,
        value_scale=value_scale,
        gamma=gamma,
    )
    detail = salience**gamma
    return gain * (ambient_gain * brightness + detail)


def _triangular_filterbank(data: np.ndarray, n_bands: int) -> np.ndarray:
    """Project a validated normalized matrix through overlapping bands."""
    n_bands = _positive_int(n_bands, name="n_bands")
    values = data
    if n_bands == 1:
        return np.mean(values, axis=1, keepdims=True)

    if values.shape[1] < n_bands:
        source_positions = np.linspace(0.0, 1.0, values.shape[1])
        target_positions = np.linspace(0.0, 1.0, n_bands)
        values = np.stack(
            [np.interp(target_positions, source_positions, row) for row in values],
            axis=0,
        )

    source_positions = np.linspace(0.0, 1.0, values.shape[1])
    centers = np.linspace(0.0, 1.0, n_bands)
    spacing = 1.0 / (n_bands - 1)
    weights = np.maximum(
        1.0 - np.abs(source_positions[None, :] - centers[:, None]) / spacing,
        0.0,
    )
    weights /= np.sum(weights, axis=1, keepdims=True)
    return np.clip(values @ weights.T, 0.0, 1.0)


def _resample_time_axis(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Map frames onto audio samples, preserving area on downsampling."""
    source_frames = data.shape[0]
    if source_frames >= n_samples:
        return _rebin_axis(data, n_samples, axis=0)
    if source_frames == 1:
        return np.repeat(data, n_samples, axis=0)

    frame_positions = np.arange(source_frames, dtype=np.float64)
    sample_positions = np.linspace(0.0, source_frames - 1.0, n_samples)
    result = np.empty((n_samples, data.shape[1]), dtype=np.float64)
    for band in range(data.shape[1]):
        result[:, band] = np.interp(sample_positions, frame_positions, data[:, band])
    return result


def _hz_to_mel(frequencies: np.ndarray | float) -> np.ndarray:
    """Convert Hz to the continuous logarithmic HTK mel scale."""
    hz = np.asarray(frequencies, dtype=np.float64)
    return 2_595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mels: np.ndarray | float) -> np.ndarray:
    """Convert continuous logarithmic HTK mel values to Hz."""
    mel = np.asarray(mels, dtype=np.float64)
    return 700.0 * np.expm1(mel * np.log(10.0) / 2_595.0)


def mel_frequencies(
    n_bands: int,
    min_freq: float = 100.0,
    max_freq: float = 2_000.0,
) -> np.ndarray:
    """Return center frequencies evenly spaced on the HTK mel scale."""
    n_bands = _positive_int(n_bands, name="n_bands")
    min_freq = _positive_float(min_freq, name="min_freq")
    max_freq = _positive_float(max_freq, name="max_freq")
    if max_freq <= min_freq:
        raise ValueError("max_freq must be greater than min_freq")
    mel_rate = np.linspace(float(_hz_to_mel(min_freq)), float(_hz_to_mel(max_freq)), n_bands)
    return _mel_to_hz(mel_rate)


def _hz_to_erb_rate(frequency: float) -> float:
    """Convert one positive frequency to the Glasberg--Moore ERB-rate axis."""
    return 21.4 * float(np.log10(1.0 + 0.00437 * frequency))


def _auditory_band_count(min_freq: float, max_freq: float) -> int:
    """Choose approximately one simultaneous voice per auditory ERB."""
    resolved_min = _positive_float(min_freq, name="min_freq")
    resolved_max = _positive_float(max_freq, name="max_freq")
    if resolved_max <= resolved_min:
        raise ValueError("max_freq must be greater than min_freq")
    erb_span = _hz_to_erb_rate(resolved_max) - _hz_to_erb_rate(resolved_min)
    return max(1, int(np.floor(erb_span + 0.5)))


def erb_frequencies(
    n_bands: int,
    min_freq: float = 100.0,
    max_freq: float = 2_000.0,
) -> np.ndarray:
    """Return center frequencies evenly spaced on the Glasberg--Moore ERB-rate axis."""
    n_bands = _positive_int(n_bands, name="n_bands")
    min_freq = _positive_float(min_freq, name="min_freq")
    max_freq = _positive_float(max_freq, name="max_freq")
    if max_freq <= min_freq:
        raise ValueError("max_freq must be greater than min_freq")
    min_erb = _hz_to_erb_rate(min_freq)
    max_erb = _hz_to_erb_rate(max_freq)
    erb_rate = np.linspace(min_erb, max_erb, n_bands)
    return (10.0 ** (erb_rate / 21.4) - 1.0) / 0.00437


def _center_frequencies(settings: _SynthesisSettings) -> np.ndarray:
    generator = mel_frequencies if settings.frequency_scale == "mel" else erb_frequencies
    return generator(settings.n_bands, settings.min_freq, settings.max_freq)


def _multitone_phases(n_bands: int) -> np.ndarray:
    """Return deterministic quadratic phases with a low multitone crest factor."""
    indices = np.arange(_positive_int(n_bands, name="n_bands"), dtype=np.float64)
    return np.pi * indices**2 / n_bands


def _temporal_salience(values: np.ndarray) -> np.ndarray:
    """Measure positive deviation in an already validated band matrix."""
    amplitudes = values
    baseline = np.median(amplitudes, axis=0, keepdims=True)
    mad = _MAD_TO_GAUSSIAN_SIGMA * np.median(
        np.abs(amplitudes - baseline),
        axis=0,
        keepdims=True,
    )
    fallback = np.std(amplitudes, axis=0, keepdims=True)
    noise_scale = np.maximum(np.where(mad > _MIN_ROBUST_SCALE, mad, fallback), _MIN_ROBUST_SCALE)
    positive_sigma = np.maximum((amplitudes - baseline) / noise_scale, 0.0)
    return -np.expm1(-positive_sigma / _SALIENCE_SATURATION_SIGMA)


def _smooth_envelopes(
    envelopes: np.ndarray,
    *,
    duration: float,
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    """Apply a frame-rate-independent one-pole attack/release envelope."""
    if attack_ms == 0 and release_ms == 0:
        return envelopes.copy()
    frame_seconds = duration / envelopes.shape[0]
    attack_coefficient = 0.0 if attack_ms == 0 else np.exp(-frame_seconds / (attack_ms / 1_000))
    release_coefficient = 0.0 if release_ms == 0 else np.exp(-frame_seconds / (release_ms / 1_000))
    smoothed = np.empty_like(envelopes)
    state = np.zeros(envelopes.shape[1], dtype=np.float64)
    for frame, target in enumerate(envelopes):
        coefficient = np.where(target > state, attack_coefficient, release_coefficient)
        state = coefficient * state + (1.0 - coefficient) * target
        smoothed[frame] = state
    return smoothed


def _a_weighting_db(frequencies: np.ndarray) -> np.ndarray:
    """Return the IEC-style A-weighting response used for gentle compensation."""
    f2 = frequencies**2
    numerator = (12_200.0**2) * f2**2
    denominator = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12_200.0**2)
    return 20.0 * np.log10(numerator / denominator) + 2.0


def _loudness_compensation_gains(
    frequencies: np.ndarray,
    *,
    limit_db: float,
) -> np.ndarray:
    """Approximately equalize perceived level with a bounded inverse A curve."""
    if limit_db == 0:
        return np.ones_like(frequencies)
    compensation_db = np.clip(-_a_weighting_db(frequencies), -limit_db, limit_db)
    return 10.0 ** (compensation_db / 20.0)


def _render_band_voice(
    sample_time: np.ndarray,
    *,
    frequency: float,
    phase: float,
    band_index: int,
    band_position: float,
    harmonic_limit_hz: float,
    settings: _SynthesisSettings,
) -> np.ndarray:
    """Render one carrier; voice choice never changes its data envelope."""
    return _render_voice(
        settings.voice.name,
        sample_time,
        frequency=frequency,
        phase=phase,
        band_index=band_index,
        band_position=band_position,
        harmonic_limit_hz=harmonic_limit_hz,
        detune_cents=settings.voice.detune_cents,
        fm_index=settings.voice.fm_index,
        chorus_rate_hz=settings.voice.chorus_rate_hz,
        chorus_depth_ms=settings.voice.chorus_depth_ms,
    )


def _mix_events(
    audio: np.ndarray,
    *,
    salience: np.ndarray,
    frequencies: np.ndarray,
    compensation_gains: np.ndarray,
    harmonic_limit_hz: float,
    event_rate_scale: float,
    settings: _SynthesisSettings,
) -> np.ndarray:
    """Apply the optional decorator after the continuous mapping is complete."""
    return _add_event_layer(
        audio,
        event_voice=settings.event.voice,
        salience=salience,
        frequencies=frequencies,
        compensation_gains=compensation_gains,
        sr=settings.sr,
        duration=settings.duration,
        threshold=settings.event.salience_threshold,
        max_events_per_second=settings.event.max_events_per_second * event_rate_scale,
        decay_ms=settings.event.decay_ms,
        level_db=settings.event.level_db,
        harmonic_limit_hz=harmonic_limit_hz,
    )


def _synthesize_prepared(
    prepared: np.ndarray,
    *,
    settings: _SynthesisSettings,
    event_rate_scale: float = 1.0,
) -> np.ndarray:
    """Synthesize one normalized matrix through one continuous ambient-detail map."""
    values = _triangular_filterbank(prepared, settings.n_bands)
    if settings.frequency_order == "descending":
        values = values[:, ::-1]

    salience = _temporal_salience(values)
    envelopes = _mapping_envelopes(
        values,
        salience=salience,
        value_scale=settings.value_scale,
        gamma=settings.gamma,
        level_db=settings.mapping_level_db,
        ambient_level_db=settings.ambient_level_db,
    )
    envelopes = _smooth_envelopes(
        envelopes,
        duration=settings.duration,
        attack_ms=settings.attack_ms,
        release_ms=settings.release_ms,
    )

    frequencies = _center_frequencies(settings)
    compensation_gains = _loudness_compensation_gains(
        frequencies,
        limit_db=settings.loudness_compensation_db,
    )
    envelopes *= compensation_gains

    n_samples = duration_to_samples(settings.duration, settings.sr)
    sample_time = np.arange(n_samples, dtype=np.float64) / settings.sr
    sample_envelopes = _resample_time_axis(envelopes, n_samples)
    audio = np.zeros(n_samples, dtype=np.float64)
    phases = _multitone_phases(settings.n_bands)
    positions = np.linspace(0.0, 1.0, settings.n_bands)
    harmonic_limit_hz = min(settings.voice.harmonic_limit_hz, settings.sr * 0.475)

    for band, (frequency, phase, position) in enumerate(zip(frequencies, phases, positions)):
        carrier = _render_band_voice(
            sample_time,
            frequency=float(frequency),
            phase=float(phase),
            band_index=band,
            band_position=float(position),
            harmonic_limit_hz=harmonic_limit_hz,
            settings=settings,
        )
        audio += sample_envelopes[:, band] * carrier

    audio /= np.sqrt(settings.n_bands)
    if settings.event.voice == "none":
        return audio
    return _mix_events(
        audio,
        salience=salience,
        frequencies=frequencies,
        compensation_gains=compensation_gains,
        harmonic_limit_hz=harmonic_limit_hz,
        event_rate_scale=event_rate_scale,
        settings=settings,
    )


def _true_peak(audio: np.ndarray) -> float:
    """Estimate inter-sample peak with 4x polyphase oversampling."""
    if not np.any(audio):
        return 0.0
    from scipy.signal import resample_poly

    oversampled = resample_poly(audio, _TRUE_PEAK_OVERSAMPLE, 1, axis=0)
    return float(np.max(np.abs(oversampled)))


def _condition_synthesized_audio(
    audio: np.ndarray,
    *,
    sr: int,
    rms_limit_dbfs: float,
    peak_limit_dbfs: float,
) -> np.ndarray:
    """Apply RMS and true-peak ceilings without lifting quiet audio."""
    conditioned = condition_audio_output(audio, sr, peak=None).astype(np.float64)
    rms = float(np.sqrt(np.mean(conditioned**2)))
    if rms == 0:
        return np.zeros_like(conditioned, dtype=np.float32)

    rms_limit = 10.0 ** (rms_limit_dbfs / 20.0)
    peak_limit = 10.0 ** (peak_limit_dbfs / 20.0)
    peak = _true_peak(conditioned)
    gain = min(1.0, rms_limit / rms)
    if peak > 0:
        gain = min(gain, peak_limit / peak)
    return (conditioned * gain).astype(np.float32)


__all__ = [
    "erb_frequencies",
    "mel_frequencies",
]
