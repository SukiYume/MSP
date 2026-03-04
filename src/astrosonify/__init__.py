"""AstroSonify - Methods for Sonifying Pulse.

Convert radio telescope time-frequency data into audible sound
using multiple sonification techniques.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .core import (
    normalize,
    del_burst,
    rebin_spectrogram,
    to_profile,
    save_audio,
)
from .hub import load_example
from .profile import profile_to_wave
from .amplitude import amplitude_modulate
from .griffinlim import griffinlim


# Lazy imports for optional-dependency methods
def astronify_sonify(*args, **kwargs):
    """Sonify using astronify library. Requires: pip install astrosonify[astronify]"""
    from .astronify_method import astronify_sonify as _impl
    return _impl(*args, **kwargs)


def hifigan(*args, **kwargs):
    """HiFi-GAN neural vocoder. Requires: pip install astrosonify[hifigan]"""
    from .hifigan import hifigan as _impl
    return _impl(*args, **kwargs)


def musicnet(*args, **kwargs):
    """WaveNet style transfer. Requires: pip install astrosonify[musicnet]"""
    from .musicnet import musicnet as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "__version__",
    "normalize",
    "del_burst",
    "rebin_spectrogram",
    "to_profile",
    "save_audio",
    "load_example",
    "astronify_sonify",
    "profile_to_wave",
    "amplitude_modulate",
    "griffinlim",
    "hifigan",
    "musicnet",
]
