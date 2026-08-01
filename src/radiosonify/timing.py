"""所有声化方法共用的物理时长、播放速度和输出整形。"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .core import _as_finite_array, _boolean, _peak_normalize, _positive_float, _positive_int

_MAX_RESAMPLE_DENOMINATOR = 10_000
_OUTPUT_FADE_SECONDS = 0.005
_MAX_OUTPUT_FADE_FRACTION = 0.05
_OUTPUT_PEAK = 0.9


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
    target = duration * repeat / speed
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
    return np.pad(audio, (0, target_samples - len(audio)))


def fit_audio_duration(
    audio: np.ndarray,
    sr: int,
    duration: float,
    *,
    preserve_pitch: bool = False,
) -> np.ndarray:
    """在保持时间顺序的前提下，把单声道音频拟合到严格时长。

    The default polyphase resampling behaves like changing playback speed, so
    duration and pitch change together. With ``preserve_pitch=True``, librosa's
    phase-vocoder time stretch is used before enforcing the exact sample count.
    """
    data = _as_finite_array(audio, name="audio", ndim=1)
    sr = _positive_int(sr, name="sr")
    target_samples = duration_to_samples(duration, sr)
    preserve_pitch = _boolean(preserve_pitch, name="preserve_pitch")

    if len(data) == target_samples:
        return _limit_resampling_peak(data, float(np.max(np.abs(data))))

    reference_peak = float(np.max(np.abs(data)))
    if reference_peak == 0:
        return np.zeros(target_samples, dtype=np.float32)
    if len(data) == 1:
        return _limit_resampling_peak(
            np.full(target_samples, data[0], dtype=np.float64),
            reference_peak,
        )

    if preserve_pitch:
        import librosa

        rate = len(data) / target_samples
        # 短瞬变若仍用 2048 点窗会被大量填零；这里选不超过输入长度的 2 次幂窗。
        n_fft = min(2048, 1 << (len(data).bit_length() - 1))
        transformed = librosa.effects.time_stretch(
            data.astype(np.float32),
            rate=rate,
            n_fft=n_fft,
            hop_length=max(1, n_fft // 4),
        )
    else:
        from scipy import signal

        # 用有理数近似控制多相滤波器规模，最后再统一裁剪/补零到精确样本数。
        ratio = Fraction(target_samples, len(data)).limit_denominator(_MAX_RESAMPLE_DENOMINATOR)
        transformed = signal.resample_poly(
            data,
            ratio.numerator,
            ratio.denominator,
            padtype="line",
        )

    transformed = _fix_sample_count(transformed, target_samples)
    return _limit_resampling_peak(transformed, reference_peak)


def condition_audio_output(audio: np.ndarray, sr: int) -> np.ndarray:
    """移除 DC、淡化边缘并恢复听感电平，且不改变样本数。"""
    data = _as_finite_array(audio, name="audio", ndim=1).copy()
    sr = _positive_int(sr, name="sr")
    input_peak = float(np.max(np.abs(data)))
    if input_peak == 0:
        return np.zeros_like(data, dtype=np.float32)
    if len(data) < 3:
        # 少于 3 个样本时不可能同时去 DC、把两端淡化为零并保留信号；
        # 优先保留有限的原始信息，仅做峰值整形。
        return (data * (_OUTPUT_PEAK / input_peak)).astype(np.float32)

    # 最终仍会峰值归一化，先缩放再中心化可避免极大有限数在求均值时溢出。
    data /= input_peak
    data -= np.mean(data)
    fade_samples = min(
        max(1, int(round(sr * _OUTPUT_FADE_SECONDS))),
        max(1, int(len(data) * _MAX_OUTPUT_FADE_FRACTION)),
    )
    # sin² 淡化在两端斜率为零，比线性淡化更不容易产生点击声。
    ramp = np.sin(np.linspace(0.0, np.pi / 2, fade_samples)) ** 2
    data[:fade_samples] *= ramp
    data[-fade_samples:] *= ramp[::-1]

    # 淡化会改变非对称短音频的加权均值。用两端为零的 sin² 窗抵消残余 DC，
    # 可同时保持首尾零值和严格样本数。
    dc_window = np.sin(np.linspace(0.0, np.pi, len(data))) ** 2
    data -= (np.mean(data) / np.mean(dc_window)) * dc_window
    data[0] = data[-1] = 0.0
    return _peak_normalize(data, peak=_OUTPUT_PEAK)


__all__ = [
    "condition_audio_output",
    "duration_to_samples",
    "fit_audio_duration",
    "target_audio_duration",
]
