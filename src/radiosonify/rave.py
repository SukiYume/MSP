"""可选的 RAVE TorchScript 音色转换后处理器。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .core import (
    _as_finite_array,
    _positive_int,
    _wav_output_path,
    require,
    save_audio,
)
from .timing import _resample_audio_rate


def _validate_rave_parameters(
    model_path: str | Path | None,
    device: str = "auto",
) -> dict[str, Any]:
    """在主声化开始前验证 RAVE 模型位置与设备字符串。"""
    if model_path is None:
        raise ValueError("RAVE postprocessing requires model_path to an exported .ts model")
    try:
        path = Path(model_path).expanduser()
    except (TypeError, ValueError) as exc:
        raise ValueError("model_path must point to an exported RAVE TorchScript file") from exc
    if not path.is_file():
        raise ValueError(f"RAVE model_path does not exist or is not a file: {path}")
    if path.suffix.lower() not in {".ts", ".pt", ".pth"}:
        raise ValueError("RAVE model_path must be an exported TorchScript .ts, .pt, or .pth file")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'")
    normalized_device = device.strip().lower()
    if (
        normalized_device not in {"auto", "cpu", "cuda", "mps"}
        and re.fullmatch(r"cuda:[0-9]+", normalized_device) is None
    ):
        raise ValueError("device must be 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'")
    # 统一 API 会在主声化之前调用本验证器；此处同时检查可选依赖和设备，
    # 避免先完成一段昂贵合成，最后才发现 RAVE 根本无法运行。
    torch = require("torch", "rave")
    _resolve_device(torch, normalized_device)
    return {"model_path": str(path), "device": normalized_device}


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"RAVE device '{requested}' was requested but CUDA is unavailable")
    if requested.startswith("cuda:"):
        device_index = int(requested.split(":", 1)[1])
        device_count = int(torch.cuda.device_count())
        if device_index >= device_count:
            raise ValueError(
                f"RAVE device '{requested}' does not exist; detected {device_count} CUDA device(s)"
            )
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("RAVE device 'mps' was requested but MPS is unavailable")
    try:
        return torch.device(requested)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid RAVE device: {requested}") from exc


def _model_int(model: Any, name: str) -> int:
    try:
        value = getattr(model, name)
        if hasattr(value, "item"):
            value = value.item()
        return _positive_int(value, name=f"RAVE model {name}")
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"exported RAVE model must expose a positive '{name}' attribute") from exc


def _load_audio(
    input_audio: np.ndarray | str | Path,
    sr: int | None,
) -> tuple[np.ndarray, int]:
    if isinstance(input_audio, (str, Path)):
        audio, file_sr = sf.read(input_audio, always_2d=False, dtype="float32")
        if sr is not None and _positive_int(sr, name="sr") != file_sr:
            raise ValueError(f"sr ({sr}) does not match WAV sample rate ({file_sr})")
        return _as_finite_array(audio, name="input_audio", ndim=(1, 2)), int(file_sr)
    if sr is None:
        raise ValueError("sr is required when input_audio is an array")
    return (
        _as_finite_array(input_audio, name="input_audio", ndim=(1, 2)),
        _positive_int(sr, name="sr"),
    )


def rave(
    input_audio: np.ndarray | str | Path,
    sr: int | None = None,
    output: str | Path | None = None,
    *,
    model_path: str | Path | None,
    device: str = "auto",
) -> tuple[np.ndarray, int]:
    """用用户提供的已导出 RAVE TorchScript 模型转换音色。

    RAVE 是审美型后处理，不是可逆的数据映射。模型文件会由
    :func:`torch.jit.load` 执行，因此只能使用可信来源的模型。官方导出模型的
    ``sr`` 和 ``n_channels`` 属性决定推理格式；单声道模型会逐声道处理 MSP 的
    立体声，从而保留声像而不把两边先混成单声道。
    """
    output_path = None if output is None else _wav_output_path(output)
    params = _validate_rave_parameters(model_path, device)
    audio, input_sr = _load_audio(input_audio, sr)
    if float(np.max(np.abs(audio))) > 1.0 + 1e-7:
        raise ValueError("input_audio must stay within [-1, 1] for RAVE inference")

    torch = require("torch", "rave")
    resolved_device = _resolve_device(torch, params["device"])
    try:
        model = torch.jit.load(params["model_path"], map_location=resolved_device).eval()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"failed to load exported RAVE model: {exc}") from exc
    model_sr = _model_int(model, "sr")
    model_channels = _model_int(model, "n_channels")

    resampled = _resample_audio_rate(audio, input_sr, model_sr)
    samples_by_channel = resampled[:, None] if resampled.ndim == 1 else resampled
    input_channels = samples_by_channel.shape[1]
    tensor = torch.from_numpy(np.asarray(samples_by_channel.T, dtype=np.float32)).to(
        resolved_device
    )

    if input_channels == model_channels:
        model_input = tensor.unsqueeze(0)
        output_layout = "direct"
    elif model_channels == 1:
        # 每个输入声道作为一个 batch item 过同一个单声道模型，避免破坏声像。
        model_input = tensor.unsqueeze(1)
        output_layout = "mono_per_channel"
    elif input_channels == 1:
        model_input = tensor.repeat(model_channels, 1).unsqueeze(0)
        output_layout = "expanded"
    else:
        raise ValueError(
            f"RAVE model expects {model_channels} channels but input has {input_channels}; "
            "only exact matches or conversion through a mono side are supported"
        )

    with torch.inference_mode():
        transformed = model.forward(model_input)
    if isinstance(transformed, (tuple, list)):
        transformed = transformed[0]
    try:
        decoded = transformed.detach().to("cpu").float().numpy()
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise ValueError("RAVE model forward() must return an audio tensor") from exc
    if decoded.ndim != 3:
        raise ValueError(
            f"RAVE model returned {decoded.ndim}D output; expected batch x channel x time"
        )

    if output_layout == "mono_per_channel":
        if decoded.shape[0] != input_channels or decoded.shape[1] != 1:
            raise ValueError("RAVE mono model returned an incompatible batch/channel shape")
        result = decoded[:, 0, :].T
    else:
        if decoded.shape[0] != 1 or decoded.shape[1] != model_channels:
            raise ValueError("RAVE model returned an incompatible batch/channel shape")
        result = decoded[0].T
    if result.shape[1] == 1:
        result = result[:, 0]
    result = _as_finite_array(result, name="RAVE output", ndim=(1, 2))
    peak = float(np.max(np.abs(result)))
    if peak > 1.0:
        result = result / peak
    result = result.astype(np.float32)

    if output_path is not None:
        save_audio(result, model_sr, output_path)
    return result, model_sr


__all__ = ["rave"]
