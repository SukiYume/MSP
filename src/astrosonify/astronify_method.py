# src/astrosonify/astronify_method.py
"""Method 0: Sonification using the astronify library."""

from __future__ import annotations

import os
import tempfile
import numpy as np
import soundfile as sf

from .core import to_profile, save_audio


def _require_astronify():
    try:
        from astropy.table import Table
        from astronify.series import SoniSeries
        return Table, SoniSeries
    except ImportError:
        raise ImportError(
            "Astronify method requires astropy and astronify. "
            "Install with: pip install astrosonify[astronify]"
        )


def astronify_sonify(
    data: np.ndarray,
    note_spacing: float = 0.01,
    time_downsample: int = 10,
    output: str | None = None,
) -> tuple[np.ndarray, int]:
    """Sonify data using the astronify library.

    Maps pulse profile intensity to pitch.

    Requires: pip install astrosonify[astronify]

    Args:
        data: 1D profile or 2D spectrogram (time x freq).
        note_spacing: Time between notes in seconds.
        time_downsample: Downsample factor for the profile.
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    Table, SoniSeries = _require_astronify()

    profile = to_profile(data, downsample=time_downsample)
    pmin, pmax = profile.min(), profile.max()
    if pmax > pmin:
        profile = (profile - pmin) / (pmax - pmin)
    else:
        profile = np.zeros_like(profile)

    data_table = Table({"time": np.arange(len(profile)), "flux": profile})
    soni = SoniSeries(data_table)
    soni.note_spacing = note_spacing
    soni.sonify()

    # Extract audio data from astronify's generated waveform via temp WAV.
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        soni.write(tmp_path)
        audio, sr = sf.read(tmp_path)
    finally:
        os.unlink(tmp_path)

    audio = audio.astype(np.float32)

    if output is not None:
        save_audio(audio, sr, output)

    return audio, sr
