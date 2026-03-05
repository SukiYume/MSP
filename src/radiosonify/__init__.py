"""RadioSonify - Methods for Sonifying Pulse.

Convert radio telescope time-frequency data into audible sound
using multiple sonification techniques.
"""

from __future__ import annotations

__version__ = "0.1.1"

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
def hifigan(*args, **kwargs):
    """HiFi-GAN neural vocoder. Requires: pip install radiosonify[hifigan]"""
    from .hifigan import hifigan as _impl
    return _impl(*args, **kwargs)


def musicnet(*args, **kwargs):
    """WaveNet style transfer. Requires: pip install radiosonify[musicnet]"""
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
    "profile_to_wave",
    "amplitude_modulate",
    "griffinlim",
    "hifigan",
    "musicnet",
]
