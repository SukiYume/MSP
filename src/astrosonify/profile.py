"""Method 1: Convert pulse profile to waveform via interpolation."""

from __future__ import annotations

import wave as wave_module
import numpy as np
from scipy import stats, signal, interpolate

from .core import to_profile, normalize, save_audio
from .hub import get_instrument_path


def _read_wave(file: str) -> np.ndarray:
    """Read a WAV file and return the first channel as int16 array."""
    with wave_module.open(file, "rb") as f:
        nchannels, sampwidth, framerate, nframes = f.getparams()[:4]
        str_data = f.readframes(nframes)
    wave_data = np.frombuffer(str_data, dtype=np.short)
    wave_data = wave_data.reshape(-1, nchannels)
    return wave_data[:, 0]


def profile_to_wave(
    data: np.ndarray,
    sr: int = 48000,
    duration: float = 10.0,
    repeat: int = 10,
    time_downsample: int | None = None,
    instrument: str | None = "violin",
    output: str | None = None,
) -> tuple[np.ndarray, int]:
    """Convert pulse profile to audible waveform.

    If data is 2D (time x freq), it is averaged along the frequency axis.
    The profile is repeated, interpolated to the target duration, and
    optionally convolved with an instrument sample.

    Args:
        data: 1D profile or 2D spectrogram (time x freq).
        sr: Sample rate in Hz.
        duration: Output audio duration in seconds.
        repeat: Number of times to tile the profile.
        time_downsample: Downsample factor. None = no downsampling.
        instrument: Instrument for convolution ('violin', 'piano', or None).
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    profile = to_profile(data, downsample=time_downsample)
    profile = np.tile(profile, repeat)

    n_samples = int(sr * duration)
    time_axis = np.arange(len(profile)) / (len(profile) - 1) * duration
    f_interp = interpolate.interp1d(time_axis, profile, kind="linear")
    wave_time = np.linspace(0, duration, n_samples, endpoint=False)
    wave_time = np.clip(wave_time, time_axis[0], time_axis[-1])
    wave_raw = f_interp(wave_time)

    wave_raw = normalize(wave_raw) * 60000
    wave_raw = wave_raw.astype(np.int16)

    if instrument is not None:
        instrument_path = get_instrument_path(instrument)
        sound = _read_wave(instrument_path)

        n_resampled = int(len(sound) * sr / 44100)
        if n_resampled > 0:
            sounds = stats.binned_statistic(
                x=np.arange(len(sound)),
                values=sound.astype(np.float64),
                bins=n_resampled,
            )[0]
            integral = np.trapezoid(sounds)
            if abs(integral) > 1e-10:
                sound_norm = sounds / integral
            else:
                sound_norm = sounds
            wave_raw = signal.convolve(wave_raw.astype(np.float64), sound_norm, mode="same")
            wave_raw = normalize(wave_raw) * 30000
            wave_raw = wave_raw.astype(np.int16)

    audio = wave_raw.astype(np.float32) / 32768.0

    if output is not None:
        save_audio(audio, sr, output)

    return audio, sr
