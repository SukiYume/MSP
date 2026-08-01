"""方法 4：把二维动态谱映射到预训练 HiFi-GAN 声码器。"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .core import (
    _as_finite_array,
    _boolean,
    _del_burst_validated,
    _normalize_validated,
    _peak_normalize,
    _positive_float,
    _positive_int,
    _rebin_spectrogram_validated,
    _validate_exposure_cut,
    _wav_output_path,
    require,
    save_audio,
)
from .hub import get_model_path


def _require_torch():
    """延迟导入 Torch，普通方法不需要加载神经运行时。"""
    return require("torch", "hifigan")


def _require_skimage():
    """延迟导入频谱缩放函数，并保留可执行的安装提示。"""
    return require("skimage.transform", "hifigan").resize


def _rescale_data(data: np.ndarray, resize_fn) -> np.ndarray:
    """把动态谱缩放到模型要求的 80 个 mel-like 频率 bin。"""
    data = resize_fn(data, (data.shape[0], 80))
    data = _normalize_validated(data)
    h, w = data.shape
    # 下面的直方图偏移和幅度范围来自发布 checkpoint 的训练预处理，不能任意改动。
    n_bins = min(max(int(h * w / 100), 1), 4096)
    a = np.histogram(data.ravel(), bins=n_bins)
    b, c = (a[1][1:] + a[1][:-1]) / 2, a[0]
    d = 0.6 - b[np.argmax(c)]
    data = (data + d) * 12 - 10.5
    data = np.clip(data, -11, 1.6)
    return data.T[np.newaxis, :, :]  # (1, 80, T)


def _prepare_spectrogram(
    spectrogram: np.ndarray,
    *,
    time_rebin: int | None,
    time_smoothing: float | None,
    clean: bool,
    exposure_cut: int,
) -> np.ndarray:
    """验证一次输入，并用免重复扫描的内部变换完成预处理。"""
    data = _as_finite_array(spectrogram, name="spectrogram", ndim=2)
    clean = _boolean(clean, name="clean")
    exposure_cut = _validate_exposure_cut(exposure_cut)

    if clean:
        data = _del_burst_validated(data, exposure_cut)

    if time_smoothing is not None:
        time_smoothing = _positive_float(time_smoothing, name="time_smoothing")
        data = gaussian_filter1d(
            data,
            sigma=time_smoothing,
            axis=0,
            mode="reflect",
        )

    if time_rebin is not None:
        data = _rebin_spectrogram_validated(data, time_rebin, None)
    return data


def _torch_load_state_dict(torch, checkpoint_path: str, device):
    """优先使用安全的 weights-only 反序列化，并兼容旧版 Torch。"""
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        warnings.warn(
            "weights_only=True not supported by this PyTorch version. "
            "Falling back to legacy torch.load(). Model checkpoints are loaded "
            "from the official Hugging Face repository (TorchLight/radiosonify).",
            UserWarning,
            stacklevel=3,
        )
        return torch.load(checkpoint_path, map_location=device)


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
    clean: bool = False,
    exposure_cut: int = 25,
    output: str | Path | None = None,
) -> tuple[np.ndarray, int]:
    """使用 HiFi-GAN 神经声码器把动态谱转换为音频。

    Requires: pip install radiosonify[hifigan]

    The spectrogram frequency axis is automatically resized to 80 mel bins.
    Model weights are downloaded from Hugging Face Hub on first use.

    Args:
        spectrogram: 2D array (time x freq).
        time_rebin: Rebin time axis. None = keep original.
        time_smoothing: Gaussian smoothing sigma along time bins. None = disabled.
            This leaves frequency-channel baselines, including narrow-band RFI,
            intact while reducing isolated time-domain granularity.
        clean: Apply del_burst cleaning.
        exposure_cut: Exposure cut for del_burst.
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    output_path = None if output is None else _wav_output_path(output)
    data = _prepare_spectrogram(
        spectrogram,
        time_rebin=time_rebin,
        time_smoothing=time_smoothing,
        clean=clean,
        exposure_cut=exposure_cut,
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
    x = _rescale_data(data, resize_fn)

    with torch.inference_mode():
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        audio = generator(x_tensor).reshape(-1).cpu().numpy()
    audio = _peak_normalize(audio, peak=0.9)

    if output_path is not None:
        save_audio(audio, sampling_rate, output_path)

    return audio, sampling_rate
