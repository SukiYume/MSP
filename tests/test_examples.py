from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np


def _load_examples_module(filename):
    module_path = Path(__file__).resolve().parents[1] / "examples" / filename
    assert module_path.exists(), f"examples/{filename} is missing"

    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unified_example_uses_explicit_duration_and_public_sonify(tmp_path, monkeypatch):
    mod = _load_examples_module("sonify_example.py")
    data = np.ones((8, 8), dtype=np.float32)
    sentinel = object()
    calls = []

    monkeypatch.setattr(mod.rs, "load_example", lambda name: data)

    def fake_sonify(source, **kwargs):
        calls.append((source, kwargs))
        return sentinel

    monkeypatch.setattr(mod.rs, "sonify", fake_sonify)
    output = tmp_path / "unified.wav"

    result = mod.sonify_example(output)

    assert result is sentinel
    source, kwargs = calls[0]
    assert source.duration == mod.EXAMPLE_DURATION_SECONDS
    assert source.name == "bundled-raw-burst"
    assert np.array_equal(source.data, data)
    assert kwargs == {
        "method": "amplitude",
        "method_params": {"freq": 880},
        "output": output,
    }


def test_unified_example_default_output_does_not_dirty_repository():
    mod = _load_examples_module("sonify_example.py")
    repository = Path(__file__).resolve().parents[1]

    assert mod.DEFAULT_OUTPUT.parent == Path(tempfile.gettempdir()) / "radiosonify"
    assert repository not in mod.DEFAULT_OUTPUT.parents
