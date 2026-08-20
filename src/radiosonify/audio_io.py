"""Waveform normalization, path validation, and WAV output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .validation import _as_finite_array, _positive_int


def _peak_normalize(audio: np.ndarray, *, peak: float = 0.95) -> np.ndarray:
    """Normalize mono or samples-by-channels audio to a requested peak."""
    if not np.isfinite(peak) or not 0 < peak <= 1:
        raise ValueError("peak must be in the interval (0, 1]")
    result = _as_finite_array(audio, name="audio", ndim=(1, 2))
    current_peak = float(np.max(np.abs(result)))
    if current_peak > 0:
        result = result * (peak / current_peak)
    return result.astype(np.float32)


def _wav_output_path(path: str | Path) -> Path:
    """Validate a WAV output path without creating directories or files."""
    try:
        output_path = Path(path)
    except (TypeError, ValueError) as exc:
        raise ValueError("path must name a WAV file") from exc
    if not output_path.name or output_path.suffix.lower() != ".wav":
        raise ValueError("path must name a file with the .wav extension")
    if output_path.exists() and not output_path.is_file():
        raise ValueError(f"path points to a directory, not a WAV file: {output_path}")
    if output_path.is_symlink() and not output_path.exists():
        raise ValueError(f"path points to a broken symbolic link: {output_path}")

    existing_parent = output_path.parent
    while not existing_parent.exists() and existing_parent.parent != existing_parent:
        existing_parent = existing_parent.parent
    if existing_parent.exists() and not existing_parent.is_dir():
        raise ValueError(f"WAV output parent is not a directory: {existing_parent}")
    return output_path


def save_audio(audio: np.ndarray, sr: int, path: str | Path) -> None:
    """Save finite, unclipped mono or multichannel audio as PCM16 WAV."""
    sr = _positive_int(sr, name="sr")
    data = _as_finite_array(audio, name="audio", ndim=(1, 2))
    if float(np.max(np.abs(data))) > 1.0 + 1e-7:
        raise ValueError("audio must stay within [-1, 1] to avoid PCM clipping")
    output_path = _wav_output_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, data, sr, format="WAV", subtype="PCM_16")


__all__ = ["save_audio"]
