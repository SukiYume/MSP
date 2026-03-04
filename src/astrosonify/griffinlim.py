"""Method 3: Griffin-Lim phase reconstruction vocoder."""

from __future__ import annotations

import copy
import numpy as np
import librosa
from scipy import signal as scipy_signal

from .core import rebin_spectrogram, del_burst, save_audio


def _mel_to_linear_matrix(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    m = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))


def _griffin_lim(
    spectrogram: np.ndarray,
    n_iter: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> np.ndarray:
    X_best = copy.deepcopy(spectrogram)
    for _ in range(n_iter):
        X_t = librosa.istft(X_best, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hann")
        est = librosa.stft(X_t, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = librosa.istft(X_best, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hann")
    return np.real(X_t)


def griffinlim(
    spectrogram: np.ndarray,
    sr: int = 48000,
    n_iter: int = 200,
    n_mels: int = 512,
    n_fft: int = 4096,
    frame_length: float = 0.04,
    preemphasis: float = 0.97,
    max_db: float = 100.0,
    ref_db: float = 20.0,
    time_rebin: int | None = None,
    freq_rebin: int | None = None,
    clean: bool = False,
    exposure_cut: int = 25,
    output: str | None = None,
) -> tuple[np.ndarray, int]:
    """Reconstruct audio from spectrogram using Griffin-Lim algorithm.

    Treats the input 2D array as a mel-spectrogram and reconstructs
    audio by iteratively estimating phase information.

    Args:
        spectrogram: 2D array (time x freq).
        sr: Sample rate in Hz.
        n_iter: Number of Griffin-Lim iterations.
        n_mels: Number of mel filter banks.
        n_fft: FFT size.
        frame_length: Frame length in seconds.
        preemphasis: Pre-emphasis coefficient. 0 to disable.
        max_db: Maximum dB for denormalization.
        ref_db: Reference dB for denormalization.
        time_rebin: Rebin time axis. None = auto.
        freq_rebin: Rebin freq axis. None = auto to n_mels.
        clean: Apply del_burst cleaning first.
        exposure_cut: Exposure cut for del_burst.
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    if spectrogram.ndim != 2:
        raise ValueError("griffinlim requires 2D spectrogram input")

    data = spectrogram.astype(np.float64)

    if clean:
        data = del_burst(data, exposure_cut=exposure_cut)

    target_freq = freq_rebin if freq_rebin is not None else n_mels
    data = rebin_spectrogram(data, time_bins=time_rebin, freq_bins=target_freq)

    win_length = int(sr * frame_length)
    hop_length = win_length // 4

    mel = data.T
    mel = (np.clip(mel, 0, 1) * max_db) - max_db + ref_db
    mel = np.power(10.0, mel * 0.05)

    m = _mel_to_linear_matrix(sr, n_fft, data.shape[1])
    mag = np.dot(m, mel)

    wav = _griffin_lim(mag, n_iter, n_fft, hop_length, win_length)

    if preemphasis > 0:
        wav = scipy_signal.lfilter([1], [1, -preemphasis], wav)

    wav, _ = librosa.effects.trim(wav)
    audio = wav.astype(np.float32)

    if output is not None:
        save_audio(audio, sr, output)

    return audio, sr
