"""Method 5: WaveNet-based music style transfer."""

from __future__ import annotations

import pathlib
import platform
import numpy as np
from scipy.io import wavfile

from .core import save_audio
from .hub import get_model_path


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "MusicNet method requires PyTorch. "
            "Install with: pip install astrosonify[musicnet]"
        )


def _require_tqdm():
    try:
        import tqdm
        return tqdm
    except ImportError:
        raise ImportError(
            "MusicNet method requires tqdm. "
            "Install with: pip install astrosonify[musicnet]"
        )


STYLE_NAMES = {
    0: "Accompaniment Violin (Beethoven)",
    1: "Solo Cello (Bach)",
    2: "Solo Piano (Bach)",
    3: "Solo Piano (Beethoven)",
    4: "String Quartet (Beethoven)",
    5: "Organ Quintet (Cambini)",
}


def musicnet(
    input_audio,
    decoder_id: int = 2,
    checkpoint_type: str = "bestmodel",
    sr: int = 48000,
    batch_size: int = 1,
    split_size: int = 20,
    output: str | None = None,
) -> tuple[np.ndarray, int]:
    """Apply music style transfer using WaveNet encoder-decoder.

    Requires: pip install astrosonify[musicnet]
    Requires CUDA GPU for inference.

    Args:
        input_audio: Path to WAV file, or 1D numpy audio array.
        decoder_id: Style decoder ID (0-5). See STYLE_NAMES for mapping.
        checkpoint_type: 'bestmodel' or 'lastmodel'.
        sr: Sample rate for loading/output.
        batch_size: Batch size for inference.
        split_size: Split size for autoregressive generation.
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    torch = _require_torch()
    tqdm_mod = _require_tqdm()
    import librosa

    if decoder_id not in range(6):
        raise ValueError(f"decoder_id must be 0-5. Available styles: {STYLE_NAMES}")
    if checkpoint_type not in ("bestmodel", "lastmodel"):
        raise ValueError("checkpoint_type must be 'bestmodel' or 'lastmodel'")

    # Handle Windows PosixPath issue with torch.load
    posix_backup = None
    if platform.system() == "Windows":
        posix_backup = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath

    try:
        # Download model files
        checkpoint_file = f"{checkpoint_type}_{decoder_id}.pth"
        checkpoint_path = get_model_path("musicnet", checkpoint_file)
        args_path = get_model_path("musicnet", "args.pth")

        from .models.musicnet import wavenet_models
        from .models.musicnet.utils import mu_law, inv_mu_law
        from .models.musicnet.wavenet import WaveNet
        from .models.musicnet.wavenet_generator import WavenetGenerator

        model_args = torch.load(args_path, map_location="cpu")[0]

        encoder = wavenet_models.Encoder(model_args)
        state = torch.load(checkpoint_path, map_location="cpu")
        encoder.load_state_dict(state["encoder_state"])
        encoder.eval()
        encoder = encoder.cuda()

        decoder = WaveNet(model_args)
        decoder.load_state_dict(state["decoder_state"])
        decoder.eval()
        decoder = decoder.cuda()
        decoder = WavenetGenerator(decoder, batch_size=batch_size, wav_freq=sr)

        # Load audio
        if isinstance(input_audio, (str, pathlib.Path)):
            data, _ = librosa.load(str(input_audio), sr=sr)
        else:
            data = np.asarray(input_audio, dtype=np.float32)

        data = mu_law(data)
        xs = torch.stack([torch.tensor(data).unsqueeze(0).float().cuda()]).contiguous()

        with torch.no_grad():
            zz = torch.cat([encoder(xs_batch) for xs_batch in torch.split(xs, batch_size)], dim=0)
            audio_res = []
            for zz_batch in torch.split(zz, batch_size):
                splits = torch.split(zz_batch, split_size, -1)
                audio_data = []
                decoder.reset()
                for cond in tqdm_mod.tqdm(splits, desc="Generating"):
                    audio_data.append(decoder.generate(cond).cpu())
                audio_data = torch.cat(audio_data, -1)
                audio_res.append(audio_data)
            audio_res = torch.cat(audio_res, dim=0)

        audio = inv_mu_law(audio_res.cpu().numpy()).squeeze()
        audio = audio.astype(np.float32)

    finally:
        if posix_backup is not None:
            pathlib.PosixPath = posix_backup

    if output is not None:
        save_audio(audio, sr, output)

    return audio, sr
