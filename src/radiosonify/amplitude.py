"""Method 2: Amplitude-modulated sine wave sonification."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from .core import save_audio, to_profile


def amplitude_modulate(
    data: np.ndarray,
    sr: int = 48000,
    duration: float = 2.0,
    freq: float = 1000.0,
    time_downsample: int | None = None,
    output: str | None = None,
) -> tuple[np.ndarray, int]:
    """Map pulse profile amplitude to loudness of a sine wave.

    Args:
        data: 1D profile or 2D spectrogram (time x freq).
        sr: Sample rate in Hz.
        duration: Output audio duration in seconds.
        freq: Carrier sine wave frequency in Hz.
        time_downsample: Downsample factor. None = no downsampling.
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    profile = to_profile(data, downsample=time_downsample)
    profile = np.log10(
        (profile - profile.min()) / (profile.max() - profile.min() + 1e-10) + 1
    )

    if duration <= 0:
        raise ValueError("duration must be > 0")

    n_samples = int(sr * duration)
    if n_samples <= 0:
        raise ValueError("duration and sr produce zero output samples")

    t_orig = np.linspace(0.0, duration, len(profile), endpoint=False)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)
    f_interp = interp1d(t_orig, profile, bounds_error=False, fill_value=(profile[0], profile[-1]))
    envelope = f_interp(t)

    carrier = np.sin(2.0 * np.pi * freq * t)
    audio = envelope * carrier

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    audio = audio.astype(np.float32)

    if output is not None:
        save_audio(audio, sr, output)

    return audio, sr
