import subprocess
import sys


def _run_isolated_import(code):
    return subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_plain_import_does_not_load_optional_or_heavy_backends():
    output = _run_isolated_import(
        """
import sys
import radiosonify
heavy = [
    "librosa",
    "huggingface_hub",
    "scipy",
    "radiosonify.profile",
    "radiosonify.griffinlim",
    "radiosonify.hifigan",
    "radiosonify.hub",
    "radiosonify.musicnet",
    "radiosonify._events",
    "radiosonify._perceptual",
    "radiosonify._voices",
    "radiosonify.erb",
    "radiosonify.rave",
    "radiosonify.spatial",
]
print([name for name in heavy if name in sys.modules])
"""
    )

    assert output == "[]"


def test_same_named_submodules_keep_standard_import_semantics():
    output = _run_isolated_import(
        """
from types import ModuleType
import radiosonify
import radiosonify.griffinlim as griffinlim_module
import radiosonify.hifigan as hifigan_module
import radiosonify.musicnet as musicnet_module
import radiosonify.rave as rave_module
for name, module in (
    ("griffinlim", griffinlim_module),
    ("hifigan", hifigan_module),
    ("musicnet", musicnet_module),
    ("rave", rave_module),
):
    print(name, isinstance(module, ModuleType), getattr(radiosonify, name) is module)
for alias in (
    "del_burst",
    "rebin_spectrogram",
    "profile_to_wave",
    "amplitude_modulate",
    "erb_sonify",
    "spatial_sonify",
    "griffinlim_reconstruct",
    "hifigan_vocode",
    "musicnet_transform",
    "rave_transform",
):
    print(alias, hasattr(radiosonify, alias))
"""
    )

    assert output.splitlines() == [
        "griffinlim True True",
        "hifigan True True",
        "musicnet True True",
        "rave True True",
        "del_burst False",
        "rebin_spectrogram False",
        "profile_to_wave False",
        "amplitude_modulate False",
        "erb_sonify False",
        "spatial_sonify False",
        "griffinlim_reconstruct False",
        "hifigan_vocode False",
        "musicnet_transform False",
        "rave_transform False",
    ]


def test_importing_hub_does_not_mutate_hugging_face_environment():
    output = _run_isolated_import(
        """
import os
os.environ.pop("HF_HUB_DISABLE_SYMLINKS_WARNING", None)
import radiosonify.hub
print(os.environ.get("HF_HUB_DISABLE_SYMLINKS_WARNING"))
"""
    )

    assert output == "None"
