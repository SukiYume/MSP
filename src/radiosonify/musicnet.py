"""Method 5: WaveNet-based music style transfer."""

from __future__ import annotations

import logging
import pathlib
import pickle
import warnings
import numpy as np
from pathlib import Path

from .core import save_audio, require
from .hub import get_model_path

_logger = logging.getLogger(__name__)


def _require_torch():
    return require("torch", "musicnet")


def _require_tqdm():
    return require("tqdm", "musicnet")


STYLE_NAMES = {
    0: "Accompaniment Violin (Beethoven)",
    1: "Solo Cello (Bach)",
    2: "Solo Piano (Bach)",
    3: "Solo Piano (Beethoven)",
    4: "String Quartet (Beethoven)",
    5: "Organ Quintet (Cambini)",
}


class _PathCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return pathlib.PurePosixPath
        return super().find_class(module, name)


class _PathCompatPickleModule:
    Unpickler = _PathCompatUnpickler


def _load_checkpoint(torch, checkpoint_path: str):
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        warnings.warn(
            "weights_only=True not supported by this PyTorch version. "
            "Falling back to legacy torch.load(). Model checkpoints are loaded "
            "from the official Hugging Face repository (TorchLight/radiosonify).",
            UserWarning,
            stacklevel=3,
        )
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            pickle_module=_PathCompatPickleModule,
        )


def musicnet(
    input_audio: str | Path | np.ndarray,
    decoder_id: int = 2,
    checkpoint_type: str = "bestmodel",
    sr: int = 48000,
    batch_size: int = 1,
    split_size: int = 20,
    num_threads: int | None = 1,
    output: str | None = None,
) -> tuple[np.ndarray, int]:
    """Apply music style transfer using WaveNet encoder-decoder.

    Requires: pip install radiosonify[musicnet]
    Runs on CPU or CUDA (CUDA recommended for speed).

    Args:
        input_audio: Path to WAV file, or 1D numpy audio array.
        decoder_id: Style decoder ID (0-5). See STYLE_NAMES for mapping.
        checkpoint_type: 'bestmodel' or 'lastmodel'.
        sr: Sample rate for loading/output.
        batch_size: Batch size for inference.
        split_size: Split size for autoregressive generation.
        num_threads: CPU threads for decoder. None = keep current. Default 1.
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    torch = _require_torch()
    _require_tqdm()
    import librosa

    if decoder_id not in range(6):
        raise ValueError(f"decoder_id must be 0-5. Available styles: {STYLE_NAMES}")
    if checkpoint_type not in ("bestmodel", "lastmodel"):
        raise ValueError("checkpoint_type must be 'bestmodel' or 'lastmodel'")

    if torch.cuda.is_available():
        enc_device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        _logger.info("encoder: cuda (%s)", gpu_name)
    else:
        enc_device = torch.device("cpu")
        _logger.info("encoder: cpu (CUDA not available)")
    dec_device = torch.device("cpu")
    _logger.info("decoder: cpu")

    checkpoint_file = f"{checkpoint_type}_{decoder_id}.pth"
    checkpoint_path = get_model_path("musicnet", checkpoint_file)
    args_path = get_model_path("musicnet", "args.json")

    from .models.musicnet import wavenet_models
    from .models.musicnet.utils import mu_law, inv_mu_law
    from .models.musicnet.wavenet import WaveNet
    from .models.musicnet.wavenet_generator import WavenetGenerator

    import json
    import argparse
    with open(args_path) as f:
        args_data = json.load(f)
    model_args = argparse.Namespace(**args_data["args"])

    encoder = wavenet_models.Encoder(model_args)
    state = _load_checkpoint(torch, checkpoint_path)
    encoder.load_state_dict(state["encoder_state"])
    encoder.eval()
    encoder = encoder.to(enc_device)

    decoder = WaveNet(model_args)
    decoder.load_state_dict(state["decoder_state"])
    decoder.eval()
    decoder = decoder.to(dec_device)
    decoder = WavenetGenerator(decoder, batch_size=batch_size, wav_freq=sr)

    # Load audio
    if isinstance(input_audio, (str, pathlib.Path)):
        data, _ = librosa.load(str(input_audio), sr=sr)
    else:
        data = np.asarray(input_audio, dtype=np.float32)

    import tqdm as _tqdm_mod

    data = mu_law(data)
    duration_sec = len(data) / sr
    _logger.info("input audio: %d samples (%.2fs @ %d Hz)", len(data), duration_sec, sr)

    xs = torch.stack([torch.tensor(data).unsqueeze(0).float().to(enc_device)]).contiguous()

    prev_threads = torch.get_num_threads()

    with torch.inference_mode():
        zz = torch.cat([encoder(xs_batch) for xs_batch in torch.split(xs, batch_size)], dim=0)
        zz = zz.to(dec_device)

        del encoder, state, xs
        if enc_device.type == "cuda":
            torch.cuda.empty_cache()

        enc_steps = zz.size(2)
        n_splits = (enc_steps + split_size - 1) // split_size
        est_sec = enc_steps * 0.8
        _logger.info(
            "encoding shape: %s -> %d steps / %d splits (est. %.0fs)",
            tuple(zz.shape), enc_steps, n_splits, est_sec,
        )

        if num_threads is not None:
            torch.set_num_threads(num_threads)
        audio_res = []
        for zz_batch in torch.split(zz, batch_size):
            splits = torch.split(zz_batch, split_size, -1)
            total_steps = zz_batch.size(2)
            audio_data = []
            decoder.reset()
            with _tqdm_mod.tqdm(total=total_steps, desc="Generating", unit="step") as pbar:
                for cond in splits:
                    audio_data.append(decoder.generate(cond, pbar=pbar).cpu())
            audio_data = torch.cat(audio_data, -1)
            audio_res.append(audio_data)
        audio_res = torch.cat(audio_res, dim=0)

    torch.set_num_threads(prev_threads)

    audio = inv_mu_law(audio_res.cpu().numpy()).squeeze()
    audio = audio.astype(np.float32)

    if output is not None:
        save_audio(audio, sr, output)

    return audio, sr
