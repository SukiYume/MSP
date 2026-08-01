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
]
print([name for name in heavy if name in sys.modules])
"""
    )

    assert output == "[]"


def test_top_level_lazy_functions_survive_submodule_first_imports():
    output = _run_isolated_import(
        """
import importlib
import radiosonify
for name in ("griffinlim", "hifigan", "musicnet"):
    importlib.import_module(f"radiosonify.{name}")
    print(name, callable(getattr(radiosonify, name)))
"""
    )

    assert output.splitlines() == [
        "griffinlim True",
        "hifigan True",
        "musicnet True",
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
