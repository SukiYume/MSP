"""方法 5：以音频为输入的 MusicNet/WaveNet 风格后处理。"""

from __future__ import annotations

import json
import logging
from argparse import Namespace
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import TypedDict

import numpy as np

from .audio_io import _peak_normalize, _wav_output_path, save_audio
from .hub import get_model_path
from .runtime import _temporary_torch_seed, require
from .timing import _resample_audio_rate
from .validation import _as_finite_array, _nonnegative_int, _positive_int

_logger = logging.getLogger(__name__)
_ENCODER_POOL = 800
_MUSICNET_SAMPLE_RATE = 16_000


class _MusicNetParameters(TypedDict):
    decoder_id: int
    checkpoint_type: str
    split_size: int
    num_threads: int | None
    seed: int | None


def _require_torch():
    """延迟导入 Torch，避免普通声化路径初始化神经运行时。"""
    return require("torch", "musicnet")


def _require_tqdm():
    """延迟导入进度条模块。"""
    return require("tqdm", "musicnet")


STYLE_NAMES = {
    0: "Accompaniment Violin (Beethoven)",
    1: "Solo Cello (Bach)",
    2: "Solo Piano (Bach)",
    3: "Solo Piano (Beethoven)",
    4: "String Quartet (Beethoven)",
    5: "Organ Quintet (Cambini)",
}


def _load_checkpoint(torch, checkpoint_path: str):
    """按受支持的 Torch 契约仅加载权重，拒绝任意 pickle 对象。"""
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def _validate_audio_input(data: np.ndarray) -> np.ndarray:
    """按 MusicNet 固定编码器契约校验单声道音频。"""
    result = _as_finite_array(data, name="input_audio", ndim=1)
    if len(result) < _ENCODER_POOL:
        raise ValueError(
            f"input_audio must contain at least {_ENCODER_POOL} samples for the pretrained encoder"
        )
    if float(np.max(np.abs(result))) > 1.0 + 1e-7:
        raise ValueError("input_audio must stay within [-1, 1]")
    return result


def _validate_musicnet_parameters(
    *,
    decoder_id: int,
    checkpoint_type: str,
    split_size: int,
    num_threads: int | None,
    seed: int | None,
) -> _MusicNetParameters:
    """集中校验 API 和底层函数共用的 MusicNet 控制参数。"""
    if (
        isinstance(decoder_id, (bool, np.bool_))
        or not isinstance(decoder_id, (int, np.integer))
        or int(decoder_id) not in STYLE_NAMES
    ):
        raise ValueError(f"decoder_id must be 0-5. Available styles: {STYLE_NAMES}")
    if not isinstance(checkpoint_type, str) or checkpoint_type not in {
        "bestmodel",
        "lastmodel",
    }:
        raise ValueError("checkpoint_type must be 'bestmodel' or 'lastmodel'")

    validated_threads = (
        None if num_threads is None else _positive_int(num_threads, name="num_threads")
    )
    validated_seed = None if seed is None else _nonnegative_int(seed, name="seed")
    return {
        "decoder_id": int(decoder_id),
        "checkpoint_type": checkpoint_type,
        "split_size": _positive_int(split_size, name="split_size"),
        "num_threads": validated_threads,
        "seed": validated_seed,
    }


def _preflight_musicnet(
    *,
    input_channels: int,
    input_sample_rate: int,
    input_samples: int,
    decoder_id: int,
    checkpoint_type: str,
    split_size: int,
    num_threads: int | None,
    seed: int | None,
) -> None:
    """Resolve optional dependencies and pinned assets before primary synthesis."""
    if input_channels != 1:
        raise ValueError("MusicNet accepts mono primary audio")
    primary_sr = _positive_int(input_sample_rate, name="primary audio sample rate")
    primary_samples = _positive_int(input_samples, name="primary audio samples")
    model_samples = max(
        1,
        round(Fraction(primary_samples * _MUSICNET_SAMPLE_RATE, primary_sr)),
    )
    if model_samples < _ENCODER_POOL:
        duration_ms = 1_000.0 * primary_samples / primary_sr
        raise ValueError(
            f"MusicNet requires at least {_ENCODER_POOL} samples at "
            f"{_MUSICNET_SAMPLE_RATE} Hz after resampling; the planned primary audio "
            f"provides {model_samples} samples ({duration_ms:.3f} ms)"
        )
    del split_size, num_threads, seed
    _require_torch()
    _require_tqdm()
    get_model_path("musicnet", f"{checkpoint_type}_{decoder_id}.pth")
    args_path = get_model_path("musicnet", "args.json")
    _load_model_args(args_path)


def _pad_for_encoder(data: np.ndarray) -> tuple[np.ndarray, int]:
    """Pad the final partial 800-sample frame and retain the exact crop length."""
    original_samples = len(data)
    remainder = original_samples % _ENCODER_POOL
    if remainder == 0:
        return data, original_samples
    padded = np.pad(data, (0, _ENCODER_POOL - remainder), mode="constant")
    return padded, original_samples


def _load_audio_input(input_audio: str | Path | np.ndarray, sr: int) -> np.ndarray:
    """加载音频，并按预训练模型的固定 16 kHz 契约重采样。"""
    sr = _positive_int(sr, name="sr")
    if not isinstance(input_audio, (str, Path)):
        data = _as_finite_array(np.asarray(input_audio), name="input_audio", ndim=1)
        if float(np.max(np.abs(data))) > 1.0 + 1e-7:
            raise ValueError("input_audio must stay within [-1, 1]")
        if sr != _MUSICNET_SAMPLE_RATE:
            data = _resample_audio_rate(data, sr, _MUSICNET_SAMPLE_RATE)
        return _validate_audio_input(data)

    input_path = Path(input_audio)
    if not input_path.is_file():
        raise FileNotFoundError(f"input audio file not found: {input_path}")

    import librosa

    # 文件自带采样率元数据，因此直接由 librosa 转到模型的原生采样率。
    data, _ = librosa.load(str(input_path), sr=_MUSICNET_SAMPLE_RATE, mono=True)
    return _validate_audio_input(data)


@contextmanager
def _temporary_num_threads(torch, num_threads: int | None):
    """临时限制 Torch CPU 线程数，并在异常时也恢复原值。"""
    previous = torch.get_num_threads()
    try:
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        yield
    finally:
        torch.set_num_threads(previous)


def _select_devices(torch):
    """编码器优先使用 CUDA；自回归解码器保持在兼容的 CPU 路径。"""
    if torch.cuda.is_available():
        encoder_device = torch.device("cuda")
        _logger.info("encoder: cuda (%s)", torch.cuda.get_device_name(0))
    else:
        encoder_device = torch.device("cpu")
        _logger.info("encoder: cpu (CUDA not available)")
    decoder_device = torch.device("cpu")
    _logger.info("decoder: cpu")
    return encoder_device, decoder_device


def _load_model_args(args_path: str) -> Namespace:
    """读取发布 checkpoint 的模型结构参数。"""
    with open(args_path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("args"), Mapping):
        raise RuntimeError(f"invalid MusicNet model arguments: {args_path}")
    return Namespace(**payload["args"])


def _build_models(
    torch,
    *,
    model_args: Namespace,
    checkpoint_path: str,
    encoder_device,
    decoder_device,
    sr: int,
):
    """构造模型、加载权重并返回推理态编码器和流式解码器。"""
    from .models.musicnet import wavenet_models
    from .models.musicnet.wavenet import WaveNet
    from .models.musicnet.wavenet_generator import WavenetGenerator

    state = _load_checkpoint(torch, checkpoint_path)
    if not isinstance(state, Mapping) or not {"encoder_state", "decoder_state"} <= set(state):
        raise RuntimeError(f"invalid MusicNet checkpoint: {checkpoint_path}")

    encoder = wavenet_models.Encoder(model_args)
    encoder.load_state_dict(state["encoder_state"])
    encoder.eval()
    encoder = encoder.to(encoder_device)

    decoder = WaveNet(model_args)
    decoder.load_state_dict(state["decoder_state"])
    decoder.eval()
    decoder = decoder.to(decoder_device)
    return encoder, WavenetGenerator(decoder, wav_freq=sr)


def _generate_splits(decoder, conditions: Sequence, *, pbar) -> list:
    """连续生成一批条件分段；仅第一段初始化自回归队列。"""
    generated = []
    for index, condition in enumerate(conditions):
        # 后续段必须复用卷积队列，否则每个 split 边界都会产生不连续跳变。
        generated.append(decoder.generate(condition, init=index == 0, pbar=pbar).cpu())
    return generated


def _decode_latents(
    torch,
    decoder,
    latents,
    *,
    split_size: int,
    num_threads: int | None,
    tqdm_module,
):
    """按时间片解码单个音频的潜变量，并保持片段间连续。"""
    with _temporary_num_threads(torch, num_threads):
        conditions = torch.split(latents, split_size, -1)
        with tqdm_module.tqdm(
            total=latents.size(2),
            desc="Generating",
            unit="step",
        ) as pbar:
            generated = _generate_splits(decoder, conditions, pbar=pbar)
    return torch.cat(generated, -1)


def musicnet(
    input_audio: str | Path | np.ndarray,
    decoder_id: int = 2,
    checkpoint_type: str = "bestmodel",
    sr: int = _MUSICNET_SAMPLE_RATE,
    split_size: int = 20,
    num_threads: int | None = 1,
    seed: int | None = 0,
    output: str | Path | None = None,
) -> tuple[np.ndarray, int]:
    """使用 WaveNet 编码器—解码器进行音乐风格后处理。

    Requires: pip install radiosonify[musicnet]
    The encoder can use CUDA; the autoregressive decoder runs on CPU and is
    normally the dominant cost.

    Args:
        input_audio: Path to WAV file, or 1D numpy audio array.
        decoder_id: Style decoder ID (0-5). See STYLE_NAMES for mapping.
        checkpoint_type: 'bestmodel' or 'lastmodel'.
        sr: Sample rate of an array input. Files use their embedded rate.
            Inference and output always use the model's native 16 kHz rate.
        split_size: Split size for autoregressive generation.
        num_threads: CPU threads for decoder. None = keep current. Default 1.
        seed: Non-negative random seed. None keeps stochastic decoding.
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    output_path = None if output is None else _wav_output_path(output)
    input_sr = _positive_int(sr, name="sr")
    parameters = _validate_musicnet_parameters(
        decoder_id=decoder_id,
        checkpoint_type=checkpoint_type,
        split_size=split_size,
        num_threads=num_threads,
        seed=seed,
    )
    data = _load_audio_input(input_audio, input_sr)
    data, original_samples = _pad_for_encoder(data)

    torch = _require_torch()
    tqdm_module = _require_tqdm()
    encoder_device, decoder_device = _select_devices(torch)

    checkpoint_file = f"{parameters['checkpoint_type']}_{parameters['decoder_id']}.pth"
    checkpoint_path = get_model_path("musicnet", checkpoint_file)
    args_path = get_model_path("musicnet", "args.json")
    model_args = _load_model_args(args_path)

    from .models.musicnet.utils import inv_mu_law, mu_law

    data = mu_law(data)
    duration_sec = len(data) / _MUSICNET_SAMPLE_RATE
    _logger.info(
        "input audio: %d samples (%.2fs @ %d Hz)",
        len(data),
        duration_sec,
        _MUSICNET_SAMPLE_RATE,
    )

    with _temporary_torch_seed(torch, parameters["seed"]):
        encoder, decoder = _build_models(
            torch,
            model_args=model_args,
            checkpoint_path=checkpoint_path,
            encoder_device=encoder_device,
            decoder_device=decoder_device,
            sr=_MUSICNET_SAMPLE_RATE,
        )
        samples = (
            torch.as_tensor(data, dtype=torch.float32, device=encoder_device)
            .reshape(1, 1, -1)
            .contiguous()
        )

        with torch.inference_mode():
            latents = encoder(samples).to(decoder_device)
            del encoder, samples
            if encoder_device.type == "cuda":
                torch.cuda.empty_cache()

            encoder_steps = latents.size(2)
            split_size = parameters["split_size"]
            n_splits = (encoder_steps + split_size - 1) // split_size
            _logger.info(
                "encoding shape: %s -> %d steps / %d splits (est. %.0fs)",
                tuple(latents.shape),
                encoder_steps,
                n_splits,
                encoder_steps * 0.8,
            )
            decoded = _decode_latents(
                torch,
                decoder,
                latents,
                split_size=split_size,
                num_threads=parameters["num_threads"],
                tqdm_module=tqdm_module,
            )

    audio = inv_mu_law(decoded.cpu().numpy()).reshape(-1)
    if len(audio) < original_samples:
        raise RuntimeError(
            "MusicNet decoder returned fewer samples than the validated input; "
            "the checkpoint and vendored model geometry are incompatible"
        )
    audio = audio[:original_samples]
    audio = _peak_normalize(audio, peak=0.95)

    if output_path is not None:
        save_audio(audio, _MUSICNET_SAMPLE_RATE, output_path)

    return audio, _MUSICNET_SAMPLE_RATE
