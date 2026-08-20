"""所有声化方法共用的物理时长、播放速度和输出整形。"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .audio_io import _peak_normalize
from .validation import _as_finite_array, _boolean, _positive_float, _positive_int

_MAX_RESAMPLE_DENOMINATOR = 10_000
_OUTPUT_FADE_SECONDS = 0.005
_MAX_OUTPUT_FADE_FRACTION = 0.05
_OUTPUT_PEAK = 0.9


def _bounded_resample_ratio(numerator: int, denominator: int) -> Fraction:
    """Return a positive rational ratio with a bounded polyphase denominator.

    ``Fraction.limit_denominator`` rounds ratios below half the reciprocal of
    the limit to ``0/1``.  SciPy rejects that ratio.  The smallest supported
    positive ratio is sufficient because every caller subsequently enforces an
    exact target sample count.
    """
    ratio = Fraction(numerator, denominator).limit_denominator(_MAX_RESAMPLE_DENOMINATOR)
    if ratio.numerator == 0:
        return Fraction(1, _MAX_RESAMPLE_DENOMINATOR)
    return ratio


def target_audio_duration(
    data_duration: float,
    speed: float = 1.0,
    repeat: int = 1,
) -> float:
    """计算目标输出时长（秒）。

    ``repeat`` repeats the represented data span before playback-speed
    adjustment. ``speed=2`` means twice the playback speed and therefore half
    the repeated duration.
    """
    duration = _positive_float(data_duration, name="data_duration")
    speed = _positive_float(speed, name="speed")
    repeat = _positive_int(repeat, name="repeat")
    try:
        target = duration * repeat / speed
    except OverflowError as exc:
        raise ValueError(
            "data_duration * repeat / speed must produce a finite target duration"
        ) from exc
    if not np.isfinite(target):
        raise ValueError("data_duration * repeat / speed must produce a finite target duration")
    return target


def duration_to_samples(duration: float, sr: int) -> int:
    """把秒数转换为最接近的正整数样本数。"""
    duration = _positive_float(duration, name="duration")
    sr = _positive_int(sr, name="sr")
    try:
        raw_samples = duration * sr
    except OverflowError as exc:
        raise ValueError("duration and sr produce too many output samples") from exc
    if not np.isfinite(raw_samples) or raw_samples > np.iinfo(np.intp).max:
        raise ValueError("duration and sr produce too many output samples")
    samples = int(round(raw_samples))
    if samples < 1:
        raise ValueError(f"duration ({duration:g}s) is shorter than one sample at {sr} Hz")
    return samples


def duration_to_frames(duration: float, sr: int, hop_length: int) -> int:
    """Convert audio duration to the nearest positive model-frame count.

    For a waveform model that emits ``hop_length`` samples per input frame,
    this is ``round(duration * sr / hop_length)``. Keeping the conversion in
    the timing layer lets generic pipelines choose an input length without
    teaching the scientific-data loader about a particular vocoder.
    """
    duration = _positive_float(duration, name="duration")
    sr = _positive_int(sr, name="sr")
    hop_length = _positive_int(hop_length, name="hop_length")
    try:
        raw_frames = duration * sr / hop_length
    except OverflowError as exc:
        raise ValueError("duration and sr produce too many model frames") from exc
    if not np.isfinite(raw_frames) or raw_frames > np.iinfo(np.intp).max:
        raise ValueError("duration and sr produce too many model frames")
    frames = int(round(raw_frames))
    if frames < 1:
        raise ValueError(
            f"duration ({duration:g}s) is shorter than one {hop_length}-sample frame at {sr} Hz"
        )
    return frames


def _limit_resampling_peak(audio: np.ndarray, reference_peak: float) -> np.ndarray:
    """防止时长变换凭空增加峰值或引入削波。"""
    if reference_peak == 0:
        return np.zeros_like(audio, dtype=np.float32)
    # 该函数也用于等长快路径；显式复制可避免缩峰时改写调用者的
    # float64 数组，并让只读输入与其他输入具有相同语义。
    result = np.array(audio, dtype=np.float64, copy=True)
    peak = float(np.max(np.abs(result)))
    limit = min(reference_peak, 1.0)
    if peak > limit:
        result *= limit / peak
    return result.astype(np.float32)


def _fix_sample_count(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """只在尾部裁剪或补零，避免默认重采样路径为此加载 librosa。"""
    if len(audio) >= target_samples:
        return audio[:target_samples]
    padding = [(0, target_samples - len(audio))] + [(0, 0)] * (audio.ndim - 1)
    return np.pad(audio, padding)


def _phase_vocoder_stretch(
    audio: np.ndarray,
    rate: float,
    *,
    n_fft: int,
) -> np.ndarray:
    """Time-stretch librosa-layout audio without its deprecated wrapper arguments."""
    import librosa

    hop_length = max(1, n_fft // 4)
    spectrum = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    stretched_spectrum = librosa.phase_vocoder(spectrum, rate=rate)
    target_samples = round(audio.shape[-1] / rate)
    return librosa.istft(
        stretched_spectrum,
        hop_length=hop_length,
        dtype=audio.dtype,
        length=target_samples,
    )


def _polyphase_resample(
    data: np.ndarray,
    ratio: Fraction,
    target_samples: int,
) -> np.ndarray:
    """Run the shared polyphase filter and enforce its exact sample count."""
    from scipy import signal

    transformed = signal.resample_poly(
        data,
        ratio.numerator,
        ratio.denominator,
        axis=0,
        padtype="line",
    )
    return _fix_sample_count(transformed, target_samples)


def _resample_audio_rate(
    audio: np.ndarray,
    source_sr: int,
    target_sr: int,
    *,
    target_samples: int | None = None,
) -> np.ndarray:
    """Convert a sample rate while preserving duration, shape, and safe peak.

    All sample-rate conversion paths use this helper so neural postprocessors,
    instrument responses, and the unified ``output_sr`` option share the same
    exact sample-count and anti-clipping contract.
    """
    data = _as_finite_array(audio, name="audio", ndim=(1, 2))
    source_sr = _positive_int(source_sr, name="source_sr")
    target_sr = _positive_int(target_sr, name="target_sr")
    reference_peak = float(np.max(np.abs(data)))
    if source_sr == target_sr:
        return _limit_resampling_peak(data, reference_peak)

    natural_samples = max(1, round(Fraction(len(data) * target_sr, source_sr)))
    target_samples = (
        natural_samples
        if target_samples is None
        else _positive_int(target_samples, name="target_samples")
    )
    if target_samples > np.iinfo(np.intp).max:
        raise ValueError("sample rates produce too many output samples")
    if reference_peak == 0:
        return np.zeros((target_samples, *data.shape[1:]), dtype=np.float32)
    if len(data) == 1:
        transformed = np.repeat(data, target_samples, axis=0)
    else:
        ratio = _bounded_resample_ratio(target_sr, source_sr)
        transformed = _polyphase_resample(data, ratio, target_samples)
    return _limit_resampling_peak(transformed, reference_peak)


def fit_audio_duration(
    audio: np.ndarray,
    sr: int,
    duration: float,
    *,
    preserve_pitch: bool = False,
) -> np.ndarray:
    """在保持时间顺序的前提下，把单声道或多声道音频拟合到严格时长。

    The default polyphase resampling behaves like changing playback speed, so
    duration and pitch change together. With ``preserve_pitch=True``, librosa's
    phase-vocoder time stretch is used before enforcing the exact sample count.
    """
    data = _as_finite_array(audio, name="audio", ndim=(1, 2))
    sr = _positive_int(sr, name="sr")
    target_samples = duration_to_samples(duration, sr)
    preserve_pitch = _boolean(preserve_pitch, name="preserve_pitch")

    if len(data) == target_samples:
        return _limit_resampling_peak(data, float(np.max(np.abs(data))))

    reference_peak = float(np.max(np.abs(data)))
    if reference_peak == 0:
        return np.zeros((target_samples, *data.shape[1:]), dtype=np.float32)
    if len(data) == 1:
        repeated = np.repeat(data, target_samples, axis=0)
        return _limit_resampling_peak(
            repeated,
            reference_peak,
        )

    if preserve_pitch:
        rate = len(data) / target_samples
        # 短瞬变若仍用 2048 点窗会被大量填零；这里选不超过输入长度的 2 次幂窗。
        n_fft = min(2048, 1 << (len(data).bit_length() - 1))
        # librosa 把最后一轴当时间；MSP 对外统一使用 soundfile 的
        # samples x channels 布局，因此多声道时在边界转置一次。
        librosa_input = data if data.ndim == 1 else data.T
        transformed = _phase_vocoder_stretch(
            librosa_input.astype(np.float32),
            rate,
            n_fft=n_fft,
        )
        if data.ndim == 2:
            transformed = transformed.T
        transformed = _fix_sample_count(transformed, target_samples)
    else:
        # 用有理数近似控制多相滤波器规模，最后再统一裁剪/补零到精确样本数。
        ratio = _bounded_resample_ratio(target_samples, len(data))
        transformed = _polyphase_resample(data, ratio, target_samples)

    return _limit_resampling_peak(transformed, reference_peak)


def condition_audio_output(
    audio: np.ndarray,
    sr: int,
    *,
    peak: float | None = _OUTPUT_PEAK,
) -> np.ndarray:
    """Remove DC and taper edges without changing the sample count.

    The default ``peak=0.9`` retains the historical loudness normalization.
    ``peak=None`` preserves the input peak (only scaling down values above 1),
    which prevents a quiet neural-vocoder background from being amplified into
    loud broadband noise.
    """
    data = _as_finite_array(audio, name="audio", ndim=(1, 2)).copy()
    sr = _positive_int(sr, name="sr")
    if peak is not None:
        peak = _positive_float(peak, name="peak")
        if peak > 1:
            raise ValueError("peak must be in the interval (0, 1] or None")
    input_peak = float(np.max(np.abs(data)))
    if input_peak == 0:
        return np.zeros_like(data, dtype=np.float32)
    if len(data) < 3:
        # 少于 3 个样本时不可能同时去 DC、把两端淡化为零并保留信号；
        # 优先保留有限的原始信息，仅做峰值整形。
        target_peak = min(input_peak, 1.0) if peak is None else peak
        return (data * (target_peak / input_peak)).astype(np.float32)

    # 最终仍会峰值归一化，先缩放再中心化可避免极大有限数在求均值时溢出。
    data /= input_peak
    data -= np.mean(data, axis=0, keepdims=True)
    fade_samples = min(
        max(1, int(round(sr * _OUTPUT_FADE_SECONDS))),
        max(1, int(len(data) * _MAX_OUTPUT_FADE_FRACTION)),
    )
    # sin² 淡化在两端斜率为零，比线性淡化更不容易产生点击声。
    ramp = np.sin(np.linspace(0.0, np.pi / 2, fade_samples)) ** 2
    ramp_shape = (fade_samples,) + (1,) * (data.ndim - 1)
    data[:fade_samples] *= ramp.reshape(ramp_shape)
    data[-fade_samples:] *= ramp[::-1].reshape(ramp_shape)

    # 淡化会改变非对称短音频的加权均值。用两端为零的 sin² 窗抵消残余 DC，
    # 可同时保持首尾零值和严格样本数。
    dc_window = np.sin(np.linspace(0.0, np.pi, len(data))) ** 2
    residual_mean = np.mean(data, axis=0, keepdims=True)
    window_shape = (len(data),) + (1,) * (data.ndim - 1)
    data -= (residual_mean / np.mean(dc_window)) * dc_window.reshape(window_shape)
    data[0] = 0.0
    data[-1] = 0.0
    target_peak = min(input_peak, 1.0) if peak is None else peak
    return _peak_normalize(data, peak=target_peak)


__all__ = [
    "condition_audio_output",
    "duration_to_frames",
    "duration_to_samples",
    "fit_audio_duration",
    "target_audio_duration",
]
