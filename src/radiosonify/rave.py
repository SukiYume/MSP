"""可选的 RAVE TorchScript 音色转换后处理器。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .audio_io import _wav_output_path, save_audio
from .runtime import _temporary_torch_seed, require
from .timing import _resample_audio_rate
from .validation import _as_finite_array, _nonnegative_int, _positive_int


@dataclass(frozen=True)
class _RaveContract:
    input_sample_rate: int
    output_sample_rate: int
    input_channels: int
    output_channels: int


@dataclass(frozen=True)
class _RaveChannelPlan:
    mode: str
    input_channels: int


def _validate_rave_parameters(
    model_path: str | Path | None,
    device: str = "auto",
    seed: int | None = 0,
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
    return {
        "model_path": str(path),
        "device": normalized_device,
        "seed": None if seed is None else _nonnegative_int(seed, name="seed"),
    }


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
    except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"exported RAVE model must expose a positive '{name}' attribute") from exc


def _model_int_vector(model: Any, name: str, size: int) -> tuple[int, ...]:
    """Read one fixed-size positive-integer vector from TorchScript metadata."""
    try:
        value = getattr(model, name)
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "to"):
            value = value.to("cpu")
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value)
        if array.shape != (size,):
            raise ValueError
        return tuple(
            _positive_int(item, name=f"RAVE model {name}[{index}]")
            for index, item in enumerate(array.tolist())
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"exported RAVE model must expose '{name}' as {size} positive integers"
        ) from exc


def _model_contract(model: Any) -> _RaveContract:
    """Parse the standard nn~ metadata attached to an exported RAVE method."""
    sampling_rate = _model_int(model, "sampling_rate")
    input_channels, input_divider, output_channels, output_divider = _model_int_vector(
        model,
        "forward_params",
        4,
    )
    if sampling_rate % input_divider or sampling_rate % output_divider:
        raise ValueError("RAVE model sampling_rate must be divisible by both forward rate dividers")
    return _RaveContract(
        input_sample_rate=sampling_rate // input_divider,
        output_sample_rate=sampling_rate // output_divider,
        input_channels=input_channels,
        output_channels=output_channels,
    )


def _load_exported_model(torch: Any, model_path: str, device: Any) -> Any:
    """Load and evaluate one trusted exported TorchScript model."""
    try:
        return torch.jit.load(model_path, map_location=device).eval()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"failed to load exported RAVE model: {exc}") from exc


def _channel_plan(input_channels: int, contract: _RaveContract) -> _RaveChannelPlan:
    """Resolve an unambiguous input layout before running the model."""
    if input_channels == contract.input_channels:
        return _RaveChannelPlan("exact", input_channels)
    if contract.input_channels == contract.output_channels == 1:
        return _RaveChannelPlan("independent", input_channels)
    if input_channels == 1:
        return _RaveChannelPlan("expand", input_channels)
    raise ValueError(
        f"RAVE model expects {contract.input_channels} input channel(s) and produces "
        f"{contract.output_channels}, but input audio has {input_channels}; only exact "
        "input matches, mono expansion, or per-channel one-in/one-out conversion are supported"
    )


def _preflight_rave(
    *,
    input_channels: int,
    input_sample_rate: int,
    input_samples: int,
    model_path: str,
    device: str,
    seed: int | None,
) -> dict[str, Any]:
    """Inspect the real model contract and reject incompatible primary audio early."""
    del input_sample_rate, input_samples, seed
    torch = require("torch", "rave")
    _resolve_device(torch, device)
    model = _load_exported_model(torch, model_path, torch.device("cpu"))
    contract = _model_contract(model)
    _channel_plan(input_channels, contract)
    return {"_expected_contract": contract}


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


def _prepare_model_input(
    torch: Any,
    audio: np.ndarray,
    input_sr: int,
    contract: _RaveContract,
    device: Any,
) -> tuple[Any, _RaveChannelPlan]:
    """Resample and lay out audio as batch-by-channel-by-time."""
    resampled = _resample_audio_rate(audio, input_sr, contract.input_sample_rate)
    samples_by_channel = resampled[:, None] if resampled.ndim == 1 else resampled
    plan = _channel_plan(samples_by_channel.shape[1], contract)
    tensor = torch.from_numpy(np.asarray(samples_by_channel.T, dtype=np.float32)).to(device)
    if plan.mode == "exact":
        return tensor.unsqueeze(0), plan
    if plan.mode == "independent":
        return tensor.unsqueeze(1), plan
    return tensor.repeat(contract.input_channels, 1).unsqueeze(0), plan


def _decode_model_output(
    transformed: Any,
    contract: _RaveContract,
    plan: _RaveChannelPlan,
) -> np.ndarray:
    """Validate the model tensor contract and restore samples-by-channels layout."""
    if isinstance(transformed, (tuple, list)):
        if not transformed:
            raise ValueError("RAVE model forward() returned an empty output sequence")
        transformed = transformed[0]
    try:
        decoded = transformed.detach().to("cpu").float().numpy()
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise ValueError("RAVE model forward() must return an audio tensor") from exc
    if decoded.ndim != 3:
        raise ValueError(
            f"RAVE model returned {decoded.ndim}D output; expected batch x channel x time"
        )
    if plan.mode == "independent":
        if decoded.shape[:2] != (plan.input_channels, 1):
            raise ValueError("RAVE mono model returned an incompatible batch/channel shape")
        result = decoded[:, 0, :].T
    else:
        if decoded.shape[:2] != (1, contract.output_channels):
            raise ValueError("RAVE model returned an incompatible batch/channel shape")
        result = decoded[0].T
    if result.shape[1] == 1:
        result = result[:, 0]
    finite = _as_finite_array(result, name="RAVE output", ndim=(1, 2))
    peak = float(np.max(np.abs(finite)))
    return np.asarray(finite if peak <= 1.0 else finite / peak, dtype=np.float32)


def rave(
    input_audio: np.ndarray | str | Path,
    sr: int | None = None,
    output: str | Path | None = None,
    *,
    model_path: str | Path | None,
    device: str = "auto",
    seed: int | None = 0,
    _expected_contract: _RaveContract | None = None,
) -> tuple[np.ndarray, int]:
    """用用户提供的已导出 RAVE TorchScript 模型转换音色。

    RAVE 是审美型后处理，不是可逆的数据映射。模型文件会由
    :func:`torch.jit.load` 执行，因此只能使用可信来源的模型。官方导出模型的
    ``sampling_rate`` 和 ``forward_params`` 元数据决定输入、输出格式；单进单出的
    模型会逐声道处理 MSP 的立体声，从而保留声像而不把两边先混成单声道。
    """
    output_path = None if output is None else _wav_output_path(output)
    params = _validate_rave_parameters(model_path, device, seed)
    torch = require("torch", "rave")
    resolved_device = _resolve_device(torch, params["device"])
    audio, input_sr = _load_audio(input_audio, sr)
    if float(np.max(np.abs(audio))) > 1.0 + 1e-7:
        raise ValueError("input_audio must stay within [-1, 1] for RAVE inference")
    model = _load_exported_model(torch, params["model_path"], resolved_device)
    contract = _model_contract(model)
    if _expected_contract is not None and contract != _expected_contract:
        raise ValueError("RAVE model contract changed after preflight")
    model_input, channel_plan = _prepare_model_input(
        torch,
        audio,
        input_sr,
        contract,
        resolved_device,
    )

    with _temporary_torch_seed(torch, params["seed"], device=resolved_device):
        with torch.inference_mode():
            transformed = model.forward(model_input)
    result = _decode_model_output(transformed, contract, channel_plan)

    if output_path is not None:
        save_audio(result, contract.output_sample_rate, output_path)
    return result, contract.output_sample_rate


__all__ = ["rave"]
