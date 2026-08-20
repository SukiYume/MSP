"""方法 4：把二维动态谱映射到预训练 HiFi-GAN 声码器。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .core import (
    _positive_float,
    _positive_int,
    _wav_output_path,
    require,
    save_audio,
)
from .hub import get_model_path
from .preprocessing import _as_normalized_array, _resize_axis


def _require_torch():
    """延迟导入 Torch，普通方法不需要加载神经运行时。"""
    return require("torch", "hifigan")


def _require_skimage():
    """延迟导入频谱缩放函数，并保留可执行的安装提示。"""
    return require("skimage.transform", "hifigan").resize


_MODEL_SAMPLE_RATE = 22_050
_MODEL_HOP_LENGTH = 256
_MODEL_FEATURE_BINS = 80
_MODEL_BACKGROUND_POSITION = 0.6
_MODEL_LOG_SCALE = 12.0
_MODEL_LOG_OFFSET = -10.5
_MODEL_LOG_MIN = -11.0
_MODEL_LOG_MAX = 1.6


def _frame_geometry(method_params: dict) -> tuple[int, int]:
    """注册表回调：HiFi-GAN 的帧几何由 checkpoint 固定，与方法参数无关。"""
    del method_params
    return _MODEL_SAMPLE_RATE, _MODEL_HOP_LENGTH


def _rescale_data(data: np.ndarray, resize_fn) -> tuple[np.ndarray, float]:
    """把共享预处理后的强度映射到发布 checkpoint 的对数 mel 幅度域。

    resize 到 80 个模型 bin、resize 后量程恢复、直方图众数对齐和 log-mel
    范围限制共同构成这个 checkpoint 的固定输入编码，不属于科学数据预处理。
    返回 ``(model_input, histogram_offset)``；偏移量随输入分布变化，因此一并
    返回给统一 API 记入溯源。
    """
    # 80-bin resize belongs to the checkpoint adapter.  Shared preprocessing may
    # intentionally retain an intermediate scientific resolution such as 512
    # channels for baseline correction and clipping.
    data = np.asarray(
        resize_fn(data, (data.shape[0], _MODEL_FEATURE_BINS)),
        dtype=np.float64,
    )
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    if data_max == data_min:
        data = np.zeros_like(data)
    else:
        data = (data - data_min) / (data_max - data_min)

    h, w = data.shape
    # 下面的直方图偏移和幅度范围来自发布 checkpoint 的训练预处理，不能任意改动。
    # Historical code used one histogram bin per 100 matrix cells. Preserve
    # that checkpoint mapping exactly for normal inputs; max(..., 1) only makes
    # sub-100-cell diagnostic inputs well-defined.
    n_bins = max(int(h * w / 100), 1)
    a = np.histogram(data.ravel(), bins=n_bins)
    b, c = (a[1][1:] + a[1][:-1]) / 2, a[0]
    d = float(_MODEL_BACKGROUND_POSITION - b[np.argmax(c)])
    data = (data + d) * _MODEL_LOG_SCALE + _MODEL_LOG_OFFSET
    data = np.clip(data, _MODEL_LOG_MIN, _MODEL_LOG_MAX)
    return data.T[np.newaxis, :, :], d  # (1, 80, T)


def _prepare_spectrogram(
    spectrogram: np.ndarray,
    *,
    time_rebin: int | None,
    time_smoothing: float | None,
) -> np.ndarray:
    """验证统一预处理后的输入。

    统一 API 已经在预处理阶段完成时间轴重分箱和平滑，这里两个参数都会是
    ``None``；保留它们只为直接调用低层函数的场景。
    """
    data = _as_normalized_array(spectrogram, name="spectrogram", ndim=2)

    if time_smoothing is not None:
        time_smoothing = _positive_float(time_smoothing, name="time_smoothing")
        data = gaussian_filter1d(
            data,
            sigma=time_smoothing,
            axis=0,
            mode="reflect",
        )

    if time_rebin is not None:
        time_rebin = _positive_int(time_rebin, name="time_rebin")
        data = _resize_axis(data, time_rebin, axis=0)
    return data


def _torch_load_state_dict(torch, checkpoint_path: str, device):
    """Load tensor weights without permitting arbitrary pickle objects."""
    return torch.load(checkpoint_path, map_location=device, weights_only=True)


@lru_cache(maxsize=2)
def _load_generator(
    config_path: str,
    checkpoint_path: str,
    device_name: str,
):
    """按模型路径和设备缓存只读推理生成器。"""
    torch = _require_torch()
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise RuntimeError(f"invalid HiFi-GAN config object: {config_path}")

    from .models.hifigan.env import AttrDict
    from .models.hifigan.generator import Generator

    device = torch.device(device_name)
    seed = int(config.get("seed", 0))
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices.append(
            device.index if device.index is not None else torch.cuda.current_device()
        )

    # 模型构造会消耗全局随机数；fork_rng 可在退出后恢复调用者的 RNG 状态。
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        generator = Generator(AttrDict(config)).to(device)

    state_dict = _torch_load_state_dict(torch, checkpoint_path, device)
    if not isinstance(state_dict, Mapping) or "generator" not in state_dict:
        raise RuntimeError(f"invalid HiFi-GAN checkpoint: {checkpoint_path}")
    generator.load_state_dict(state_dict["generator"])
    generator.eval()
    generator.remove_weight_norm()
    sampling_rate = _positive_int(config.get("sampling_rate"), name="model sampling_rate")
    return generator, sampling_rate, device


def hifigan(
    spectrogram: np.ndarray,
    time_rebin: int | None = None,
    time_smoothing: float | None = None,
    output: str | Path | None = None,
    *,
    provenance: dict | None = None,
) -> tuple[np.ndarray, int]:
    """使用 HiFi-GAN 神经声码器把动态谱转换为音频。

    Requires: pip install radiosonify[hifigan]

    The spectrogram frequency axis is automatically resized to 80 mel bins.
    Model weights are downloaded from Hugging Face Hub on first use.

    Args:
        spectrogram: Preprocessed ``[0, 1]`` 2D array (time x feature). The
            feature width remains a scientific-preprocessing choice; this
            checkpoint adapter always resizes it internally to 80 bins.
        time_rebin: Rebin time axis. None = keep original.
        time_smoothing: Gaussian smoothing sigma along time bins. None = disabled.
            This leaves frequency-channel baselines, including narrow-band RFI,
            intact while reducing isolated time-domain granularity.
        output: Path to save WAV file. None = don't save.
        provenance: Optional dict that receives data-dependent quantities
            resolved during the call, currently ``histogram_offset``.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    output_path = None if output is None else _wav_output_path(output)
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("provenance must be a dict or None")
    data = _prepare_spectrogram(
        spectrogram,
        time_rebin=time_rebin,
        time_smoothing=time_smoothing,
    )

    torch = _require_torch()
    resize_fn = _require_skimage()

    # 先验证输入和依赖，再下载大模型；错误输入不会触发网络或磁盘写入。
    config_path = get_model_path("hifigan", "config.json")
    checkpoint_path = get_model_path("hifigan", "generator.pth")

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    generator, sampling_rate, device = _load_generator(
        config_path,
        checkpoint_path,
        device_name,
    )

    # 输入布局固定为 (batch=1, mel=80, time)。
    x, histogram_offset = _rescale_data(data, resize_fn)

    with torch.inference_mode():
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        audio = generator(x_tensor).reshape(-1).cpu().numpy()

    if output_path is not None:
        save_audio(audio, sampling_rate, output_path)

    if provenance is not None:
        provenance["histogram_offset"] = histogram_offset
    return audio, sampling_rate
