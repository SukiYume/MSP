"""Sparse, deterministic event extraction and rendering for perceptual scans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _SoundEvent:
    """One local salience peak in frame and perceptual-band coordinates."""

    frame: int
    band: int
    strength: float


def _local_peak_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    """Find 3-by-3 local maxima without importing a morphology dependency."""
    frames, bands = values.shape
    padded = np.pad(values, ((1, 1), (1, 1)), constant_values=-np.inf)
    neighbors = []
    for frame_offset in range(3):
        for band_offset in range(3):
            if frame_offset == 1 and band_offset == 1:
                continue
            neighbors.append(
                padded[
                    frame_offset : frame_offset + frames,
                    band_offset : band_offset + bands,
                ]
            )
    neighbor_maximum = np.maximum.reduce(neighbors)
    preceding_maximum = np.maximum.reduce(
        (
            padded[0:frames, 0:bands],
            padded[0:frames, 1 : bands + 1],
            padded[0:frames, 2 : bands + 2],
            padded[1 : frames + 1, 0:bands],
        )
    )
    # A lexicographic tie-break selects the top-left representative of a flat
    # peak, preventing one plateau from producing events in several rate bins.
    return (values >= threshold) & (values >= neighbor_maximum) & (values > preceding_maximum)


def _detect_events(
    salience: np.ndarray,
    *,
    duration: float,
    threshold: float,
    max_events_per_second: float,
) -> tuple[_SoundEvent, ...]:
    """Select at most one strongest local peak per temporal rate bucket."""
    values = salience
    if max_events_per_second <= 0 or not np.any(values >= threshold):
        return ()

    event_budget = int(np.floor(duration * max_events_per_second + 1e-12))
    if event_budget <= 0:
        return ()
    candidate_frames, candidate_bands = np.nonzero(_local_peak_mask(values, threshold))
    if candidate_frames.size == 0:
        return ()

    bucket_indices = np.minimum(
        candidate_frames * event_budget // values.shape[0],
        event_budget - 1,
    )
    selected: list[_SoundEvent] = []
    for bucket in np.unique(bucket_indices):
        candidates = np.flatnonzero(bucket_indices == bucket)
        strengths = values[candidate_frames[candidates], candidate_bands[candidates]]
        winner = int(candidates[int(np.argmax(strengths))])
        frame = int(candidate_frames[winner])
        band = int(candidate_bands[winner])
        selected.append(_SoundEvent(frame, band, float(values[frame, band])))
    return tuple(selected)


def _render_water_drop(
    *,
    sr: int,
    frequency: float,
    decay_ms: float,
    harmonic_limit_hz: float,
    variation_key: int = 0,
) -> np.ndarray:
    """Render a quiet impact followed by a delayed, damped bubble resonance."""
    decay_seconds = decay_ms / 1_000.0
    sample_count = max(2, int(np.ceil(decay_seconds * sr)))
    spectral_limit = min(harmonic_limit_hz, sr * 0.475)
    # A local generator gives repeated events organic micro-variation while
    # keeping an identical input byte-for-byte deterministic.
    rng = np.random.default_rng(np.uint64(variation_key & 0xFFFFFFFFFFFFFFFF))
    pitch_cents = float(rng.uniform(-7.0, 7.0))
    resonance_frequency = min(frequency * 2.0 ** (pitch_cents / 1_200.0), spectral_limit)

    desired_delay = float(rng.uniform(0.0035, 0.0065))
    delay_seconds = min(desired_delay, decay_seconds * 0.18)
    ring_start = min(int(round(delay_seconds * sr)), sample_count - 2)
    ring_time = np.arange(sample_count - ring_start, dtype=np.float64) / sr

    # Bubble sound is a compact, nearly stationary wave packet. A sub-percent
    # settling term avoids a sterile oscillator without creating an audible
    # pitch sweep.
    settling_ratio = float(rng.uniform(0.003, 0.009))
    settling_tau = max(0.0015, 2.5 / resonance_frequency)
    phase = (
        2.0
        * np.pi
        * (
            resonance_frequency * ring_time
            + resonance_frequency
            * settling_ratio
            * settling_tau
            * (1.0 - np.exp(-ring_time / settling_tau))
        )
    )
    ring = np.sin(phase)

    # A weak, rapidly damped non-spherical mode supplies a little natural
    # texture while the mapped fundamental remains the dominant pitch.
    shape_frequency = resonance_frequency * float(rng.uniform(1.42, 1.58))
    if shape_frequency <= spectral_limit:
        shape_gain = float(rng.uniform(0.035, 0.060))
        shape_phase = float(rng.uniform(0.15, 0.85))
        ring += shape_gain * np.sin(2.0 * np.pi * shape_frequency * ring_time + shape_phase)

    e_fold_cycles = float(rng.uniform(9.0, 13.0))
    ring_tau = min(
        max(0.006, e_fold_cycles / resonance_frequency),
        max(1.0 / sr, (decay_seconds - ring_start / sr) / 4.8),
    )
    ring_attack = 1.0 - np.exp(-ring_time / max(0.00035, 1.0 / sr))
    ring_envelope = ring_attack * np.exp(-ring_time / ring_tau)

    output = np.zeros(sample_count, dtype=np.float64)
    output[ring_start:] = ring * ring_envelope

    # The initial contact is a short band-limited pressure pulse. It stays well
    # below the bubble packet so it reads as water-surface texture instead of a
    # digital click or a second pitched note.
    impact_seconds = min(0.0045, max(delay_seconds * 0.8, 4.0 / sr))
    impact_count = min(max(4, int(np.ceil(impact_seconds * sr))), sample_count)
    impact_time = np.arange(impact_count, dtype=np.float64) / sr
    noise = rng.standard_normal(impact_count)
    spectrum = np.fft.rfft(noise)
    bins = np.fft.rfftfreq(impact_count, 1.0 / sr)
    center = min(max(resonance_frequency * 1.35, 650.0), spectral_limit * 0.78)
    safe_bins = np.maximum(bins, 1.0)
    log_distance = np.log2(safe_bins / max(center, 1.0))
    spectral_shape = np.exp(-0.5 * (log_distance / 0.85) ** 2)
    spectral_shape[(bins == 0.0) | (bins > spectral_limit)] = 0.0
    impact = np.fft.irfft(spectrum * spectral_shape, n=impact_count)
    impact_peak = float(np.max(np.abs(impact)))
    if impact_peak > 0.0:
        impact /= impact_peak
    impact_attack = 1.0 - np.exp(-impact_time / max(0.00012, 1.0 / sr))
    impact_envelope = impact_attack * np.exp(-impact_time / max(0.00075, 1.0 / sr))
    output[:impact_count] += 0.11 * impact * impact_envelope
    return output


def _event_variation_key(event: _SoundEvent) -> int:
    """Derive a stable renderer key from an event's discrete coordinates."""
    frame_key = (event.frame + 1) * 0x9E3779B1
    band_key = (event.band + 1) * 0x85EBCA77
    return (frame_key ^ band_key) & 0xFFFFFFFFFFFFFFFF


def _add_event_layer(
    audio: np.ndarray,
    *,
    event_voice: str,
    salience: np.ndarray,
    frequencies: np.ndarray,
    compensation_gains: np.ndarray,
    sr: int,
    duration: float,
    threshold: float,
    max_events_per_second: float,
    decay_ms: float,
    level_db: float,
    harmonic_limit_hz: float,
) -> np.ndarray:
    """Add a sparse event layer to an owned mono synthesis buffer."""
    if event_voice == "none":
        return audio
    if event_voice != "water_drop":  # Settings validation normally prevents this branch.
        raise ValueError(f"unknown event voice: {event_voice}")

    events = _detect_events(
        salience,
        duration=duration,
        threshold=threshold,
        max_events_per_second=max_events_per_second,
    )
    event_gain = 10.0 ** (level_db / 20.0)
    frame_count = salience.shape[0]
    for event in events:
        start = (
            0
            if frame_count == 1
            else int(round(event.frame * (len(audio) - 1) / (frame_count - 1)))
        )
        drop = _render_water_drop(
            sr=sr,
            frequency=float(frequencies[event.band]),
            decay_ms=decay_ms,
            harmonic_limit_hz=harmonic_limit_hz,
            variation_key=_event_variation_key(event),
        )
        stop = min(len(audio), start + len(drop))
        gain = event_gain * event.strength * float(compensation_gains[event.band])
        audio[start:stop] += gain * drop[: stop - start]
    return audio


__all__ = [
    "_SoundEvent",
    "_add_event_layer",
    "_detect_events",
    "_event_variation_key",
    "_render_water_drop",
]
