"""Method 4: HiFi-GAN neural vocoder sonification."""

from __future__ import annotations

import json
import numpy as np

from .core import rebin_spectrogram, del_burst, normalize, save_audio
from .hub import get_model_path


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "HiFi-GAN method requires PyTorch. "
            "Install with: pip install astrosonify[hifigan]"
        )


def _require_skimage():
    try:
        from skimage.transform import resize
        return resize
    except ImportError:
        raise ImportError(
            "HiFi-GAN method requires scikit-image. "
            "Install with: pip install astrosonify[hifigan]"
        )


def _rescale_data(data: np.ndarray, resize_fn) -> np.ndarray:
    """Rescale spectrogram to 80 mel bins with HiFi-GAN normalization."""
    data = resize_fn(data, (data.shape[0], 80))
    data = normalize(data)
    h, w = data.shape
    # Parameters below follow original model preprocessing convention.
    # They are preserved for compatibility with the released checkpoint.
    n_bins = min(max(int(h * w / 100), 1), 4096)
    a = np.histogram(data.ravel(), bins=n_bins)
    b, c = (a[1][1:] + a[1][:-1]) / 2, a[0]
    d = 0.6 - b[np.argmax(c)]
    data = (data + d) * 12 - 10.5
    data = np.clip(data, -11, 1.6)
    return data.T[np.newaxis, :, :]  # (1, 80, T)


def _torch_load_state_dict(torch, checkpoint_path: str, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def hifigan(
    spectrogram: np.ndarray,
    time_rebin: int | None = None,
    clean: bool = False,
    exposure_cut: int = 25,
    output: str | None = None,
) -> tuple[np.ndarray, int]:
    """Convert spectrogram to audio using HiFi-GAN neural vocoder.

    Requires: pip install astrosonify[hifigan]

    The spectrogram frequency axis is automatically resized to 80 mel bins.
    Model weights are downloaded from Hugging Face Hub on first use.

    Args:
        spectrogram: 2D array (time x freq).
        time_rebin: Rebin time axis. None = keep original.
        clean: Apply del_burst cleaning.
        exposure_cut: Exposure cut for del_burst.
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    torch = _require_torch()
    resize_fn = _require_skimage()

    if spectrogram.ndim != 2:
        raise ValueError("hifigan requires 2D spectrogram input")

    data = spectrogram.astype(np.float64)

    if clean:
        data = del_burst(data, exposure_cut=exposure_cut)

    # Keep original time_rebin logic but remove auto-trigger
    if time_rebin is not None:
        data = rebin_spectrogram(data, time_bins=time_rebin)

    # Download model files
    config_path = get_model_path("hifigan", "config.json")
    checkpoint_path = get_model_path("hifigan", "generator.pth")

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    from .models.hifigan.env import AttrDict
    from .models.hifigan.generator import Generator

    h = AttrDict(config)
    seed = int(config["seed"])
    sampling_rate = int(config["sampling_rate"])
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    generator = Generator(h).to(device)
    state_dict = _torch_load_state_dict(torch, checkpoint_path, device)
    generator.load_state_dict(state_dict["generator"])
    generator.eval()
    generator.remove_weight_norm()

    # Rescale input to (1, 80, T)
    x = _rescale_data(data, resize_fn)

    with torch.no_grad():
        x_tensor = torch.FloatTensor(x).to(device)
        audio_tensor = generator(x_tensor).squeeze()
        audio = audio_tensor.cpu().numpy()

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    audio = audio.astype(np.float32)

    sr = sampling_rate

    if output is not None:
        save_audio(audio, sr, output)

    return audio, sr
