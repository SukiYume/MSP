"""Hugging Face Hub download manager for models and example data."""

from __future__ import annotations

import os
import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = "TorchLight/astrosonify"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "astrosonify")

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


def get_data_path(filename: str) -> str:
    """Download example data file from HF Hub and return local path."""
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=f"data/{filename}",
        cache_dir=CACHE_DIR,
    )


def get_model_path(model_name: str, filename: str) -> str:
    """Download model file from HF Hub and return local path."""
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=f"models/{model_name}/{filename}",
        cache_dir=CACHE_DIR,
    )


def get_instrument_path(name: str) -> str:
    """Download instrument sample from HF Hub and return local path."""
    if name not in INSTRUMENT_MAP:
        raise ValueError(f"Unknown instrument: {name}. Available: {list(INSTRUMENT_MAP.keys())}")
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=f"instruments/{INSTRUMENT_MAP[name]}",
        cache_dir=CACHE_DIR,
    )


def load_example(name: str) -> np.ndarray:
    """Load an example dataset by name.

    Args:
        name: One of 'burst', 'raw_burst', 'parkes_burst', 'profile'.
    """
    if name not in EXAMPLE_MAP:
        raise ValueError(f"Unknown example: {name}. Available: {list(EXAMPLE_MAP.keys())}")
    path = get_data_path(EXAMPLE_MAP[name])
    return np.load(path)
