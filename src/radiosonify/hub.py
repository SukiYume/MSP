"""Hugging Face Hub download manager for models and example data."""

from __future__ import annotations

import os
import time
import numpy as np

# Suppress the Windows symlink warning from huggingface_hub; caching still works.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import hf_hub_download

REPO_ID = "TorchLight/radiosonify"
CACHE_DIR = os.environ.get(
    "RADIOSONIFY_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "radiosonify"),
)

EXAMPLE_MAP = {
    "burst": "Burst.npy",
    "raw_burst": "RawBurst.npy",
    "parkes_burst": "ParkesBurst.npy",
    "profile": "Profile.npy",
}

INSTRUMENT_MAP = {
    "violin": "vio.wav",
    "piano": "piano.wav",
}


def _download_with_context(filename: str) -> str:
    # Try local cache first to avoid unnecessary network round-trips.
    try:
        return hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            cache_dir=CACHE_DIR,
            local_files_only=True,
        )
    except Exception:
        pass

    # File not in local cache – attempt to download from the hub.
    last_error = None
    for attempt in range(2):
        try:
            return hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                cache_dir=CACHE_DIR,
            )
        except Exception as exc:  # pragma: no cover - upstream exception shapes vary.
            last_error = exc
            if attempt == 0:
                time.sleep(0.3)

    raise RuntimeError(
        f"Failed to download '{filename}' from Hugging Face repo '{REPO_ID}'. "
        "Check network connectivity, Hugging Face access permissions, and local cache integrity. "
        f"Original error: {last_error}"
    ) from last_error


def get_data_path(filename: str) -> str:
    """Download example data file from HF Hub and return local path."""
    return _download_with_context(f"data/{filename}")


def get_model_path(model_name: str, filename: str) -> str:
    """Download model file from HF Hub and return local path."""
    return _download_with_context(f"models/{model_name}/{filename}")


def get_instrument_path(name: str) -> str:
    """Download instrument sample from HF Hub and return local path."""
    if name not in INSTRUMENT_MAP:
        raise ValueError(f"Unknown instrument: {name}. Available: {list(INSTRUMENT_MAP.keys())}")
    return _download_with_context(f"instruments/{INSTRUMENT_MAP[name]}")


def load_example(name: str) -> np.ndarray:
    """Load an example dataset by name.

    Args:
        name: One of 'burst', 'raw_burst', 'parkes_burst', 'profile'.
    """
    if name not in EXAMPLE_MAP:
        raise ValueError(f"Unknown example: {name}. Available: {list(EXAMPLE_MAP.keys())}")
    path = get_data_path(EXAMPLE_MAP[name])
    return np.load(path)
