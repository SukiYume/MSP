"""方法 3：把二维动态谱视作幅度图，用 Griffin–Lim 重建相位。"""

from __future__ import annotations

import warnings
from pathlib import Path

import librosa
import numpy as np
from scipy import signal as scipy_signal

from .core import (
    _as_finite_array,
    _boolean,
    _del_burst_validated,
    _finite_float,
    _normalize_validated,
    _peak_normalize,
    _positive_float,
    _positive_int,
    _rebin_spectrogram_validated,
    _validate_exposure_cut,
    _wav_output_path,
    save_audio,
)

# ---------- 频谱变换内部步骤 ----------

_DEFAULT_FREQ_BINS = 512


def _mel_to_linear_matrix(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """构造从 mel-like 频率轴回到线性频率轴的稳定近似逆矩阵。"""
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    transpose = mel_filter.T
    overlap = mel_filter @ transpose
    overlap_sum = np.sum(overlap, axis=0)
    inverse = np.divide(
        1.0,
        overlap_sum,
        out=np.zeros_like(overlap_sum),
        where=np.abs(overlap_sum) > 1.0e-8,
    )
    # 等价于 transpose @ diag(inverse)，但不构造大型对角矩阵。
    return transpose * inverse


def _griffin_lim(
    spectrogram: np.ndarray,
    n_iter: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> np.ndarray:
    """从非负幅度谱迭代估计相位，使用确定性的零相位初值。"""
    estimate = spectrogram.copy()
    for _ in range(n_iter):
        waveform = librosa.istft(
            estimate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
        )
        stft_estimate = librosa.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        phase = stft_estimate / np.maximum(1e-8, np.abs(stft_estimate))
        estimate = spectrogram * phase
    waveform = librosa.istft(
        estimate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
    )
    return np.real(waveform)


def _validate_synthesis_settings(
    *,
    sr: int,
    n_iter: int,
    n_fft: int,
    frame_length: float,
    preemphasis: float,
    max_db: float,
    ref_db: float,
) -> tuple[int, int, int, int, int, float, float, float]:
    """集中校验 Griffin–Lim 标量参数并计算窗长、步长。"""
    sr = _positive_int(sr, name="sr")
    n_iter = _positive_int(n_iter, name="n_iter")
    n_fft = _positive_int(n_fft, name="n_fft")
    frame_length = _positive_float(frame_length, name="frame_length")
    preemphasis = _finite_float(preemphasis, name="preemphasis")
    max_db = _positive_float(max_db, name="max_db")
    ref_db = _finite_float(ref_db, name="ref_db")

    if not 0 <= preemphasis < 1:
        raise ValueError("preemphasis must be in the interval [0, 1)")
    if ref_db > max_db:
        raise ValueError("ref_db cannot exceed max_db")

    win_length = int(sr * frame_length)
    if win_length < 2:
        raise ValueError("frame_length and sr must produce at least two samples")
    if win_length > n_fft:
        raise ValueError("frame_length and sr produce a window longer than n_fft")
    hop_length = win_length // 4
    if hop_length < 1:
        raise ValueError("frame_length and sr produce a zero hop length")

    return (
        sr,
        n_iter,
        n_fft,
        win_length,
        hop_length,
        preemphasis,
        max_db,
        ref_db,
    )


def _resolve_frequency_bins(n_mels: int | None, freq_rebin: int | None) -> int | None:
    """把旧 ``n_mels`` 关键字迁移到唯一的频率分箱控制量。"""
    if n_mels is not None:
        warnings.warn(
            "n_mels is deprecated; use freq_rebin instead",
            DeprecationWarning,
            stacklevel=3,
        )
        if freq_rebin is not None:
            raise ValueError("n_mels and freq_rebin cannot be supplied together; use freq_rebin")
        return _positive_int(n_mels, name="n_mels")
    if freq_rebin is None:
        return None
    return _positive_int(freq_rebin, name="freq_rebin")


def _prepare_spectrogram(
    spectrogram: np.ndarray,
    *,
    n_fft: int,
    time_rebin: int | None,
    freq_rebin: int | None,
    clean: bool,
    exposure_cut: int,
) -> np.ndarray:
    """校验、清理、归一化并重分箱动态谱。"""
    data = _as_finite_array(spectrogram, name="spectrogram", ndim=2)
    clean = _boolean(clean, name="clean")
    exposure_cut = _validate_exposure_cut(exposure_cut)

    target_freq = min(_DEFAULT_FREQ_BINS, data.shape[1]) if freq_rebin is None else freq_rebin
    if target_freq > n_fft // 2 + 1:
        raise ValueError("frequency bins cannot exceed n_fft // 2 + 1")

    if clean:
        data = _del_burst_validated(data, exposure_cut)
    else:
        # 望远镜强度通常不是 [0, 1]；不先归一化会在 dB 映射时整体饱和。
        data = _normalize_validated(data)
    return _rebin_spectrogram_validated(data, time_rebin, target_freq)


def griffinlim(
    spectrogram: np.ndarray,
    sr: int = 48000,
    n_iter: int = 64,
    n_mels: int | None = None,
    n_fft: int = 4096,
    frame_length: float = 0.04,
    preemphasis: float = 0.0,
    max_db: float = 100.0,
    ref_db: float = 20.0,
    time_rebin: int | None = None,
    freq_rebin: int | None = None,
    clean: bool = False,
    exposure_cut: int = 25,
    output: str | Path | None = None,
) -> tuple[np.ndarray, int]:
    """使用 Griffin–Lim 从动态谱重建音频。

    Treats the input 2D array as a mel-spectrogram and reconstructs
    audio by iteratively estimating phase information.

    Args:
        spectrogram: 2D array (time x freq).
        sr: Sample rate in Hz.
        n_iter: Number of Griffin-Lim iterations. The default 64 balances
            convergence against the mel-to-linear approximation's error floor.
        n_mels: Deprecated alias for ``freq_rebin``.
        n_fft: FFT size.
        frame_length: Frame length in seconds.
        preemphasis: Optional de-emphasis coefficient used only for deliberate
            tonal coloring. The default 0 disables it because the input has no
            paired pre-emphasis stage.
        max_db: Maximum dB for denormalization.
        ref_db: Reference dB for denormalization.
        time_rebin: Rebin time axis. None = auto.
        freq_rebin: Target frequency-bin count. None = keep at most 512 bins.
        clean: Apply del_burst cleaning first.
        exposure_cut: Exposure cut for del_burst.
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    output_path = None if output is None else _wav_output_path(output)
    (
        sr,
        n_iter,
        n_fft,
        win_length,
        hop_length,
        preemphasis,
        max_db,
        ref_db,
    ) = _validate_synthesis_settings(
        sr=sr,
        n_iter=n_iter,
        n_fft=n_fft,
        frame_length=frame_length,
        preemphasis=preemphasis,
        max_db=max_db,
        ref_db=ref_db,
    )
    resolved_freq_rebin = _resolve_frequency_bins(n_mels, freq_rebin)
    data = _prepare_spectrogram(
        spectrogram,
        n_fft=n_fft,
        time_rebin=time_rebin,
        freq_rebin=resolved_freq_rebin,
        clean=clean,
        exposure_cut=exposure_cut,
    )

    if float(np.ptp(data)) == 0:
        silent_samples = max(1, hop_length * max(data.shape[0] - 1, 1))
        audio = np.zeros(silent_samples, dtype=np.float32)
    else:
        # 将归一化强度解释为相对 dB，再恢复为线性幅度。
        mel = (data.T * max_db) - max_db + ref_db
        mel = np.power(10.0, mel * 0.05)
        magnitude = _mel_to_linear_matrix(sr, n_fft, data.shape[1]) @ mel
        waveform = _griffin_lim(magnitude, n_iter, n_fft, hop_length, win_length)

        if preemphasis > 0:
            waveform = scipy_signal.lfilter([1], [1, -preemphasis], waveform)
        # 首尾低能量区仍代表观测时间轴；自动 trim 会移动事件并破坏绝对时间位置。
        if waveform.size == 0:
            waveform = np.zeros(1, dtype=np.float64)
        audio = _peak_normalize(waveform, peak=0.95)

    if output_path is not None:
        save_audio(audio, sr, output_path)

    return audio, sr
