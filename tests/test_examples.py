from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_examples_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "reproduce_legacy_wavs.py"
    assert module_path.exists(), "examples/reproduce_legacy_wavs.py is missing"

    spec = importlib.util.spec_from_file_location("reproduce_legacy_wavs", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_legacy_output_paths_matches_legacy_filenames(tmp_path):
    mod = _load_examples_module()

    outputs = mod.build_legacy_output_paths(tmp_path)

    assert outputs[0].name == "Audio.wav"
    assert outputs[1].name == "Audio.wav"
    assert outputs[2].name == "Audio.wav"
    assert outputs[3].name == "Audio.wav"
    assert outputs[4].name == "RawBurst_Generated.wav"
    assert outputs[5].name == "MusicNet_Converted.wav"
    assert outputs[0].parent.name == "method0_astronify"
    assert outputs[5].parent.name == "method5_musicnet"


def test_reproduce_legacy_wavs_generates_all_outputs(tmp_path, monkeypatch):
    mod = _load_examples_module()

    called = []

    def _writer(output):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"RIFF")

    monkeypatch.setattr(mod.asf, "load_example", lambda name: np.ones((8, 8), dtype=np.float32))

    monkeypatch.setattr(
        mod.asf,
        "astronify_sonify",
        lambda data, note_spacing, time_downsample, output: (called.append("m0"), _writer(output), (np.zeros(8), 48000))[-1],
    )
    monkeypatch.setattr(
        mod.asf,
        "profile_to_wave",
        lambda data, sr, duration, repeat, instrument, output: (called.append("m1"), _writer(output), (np.zeros(8), sr))[-1],
    )
    monkeypatch.setattr(
        mod.asf,
        "amplitude_modulate",
        lambda data, sr, duration, freq, output: (called.append("m2"), _writer(output), (np.zeros(8), sr))[-1],
    )
    monkeypatch.setattr(
        mod.asf,
        "griffinlim",
        lambda data, sr, n_iter, n_mels, n_fft, time_rebin, freq_rebin, output: (called.append("m3"), _writer(output), (np.zeros(8), sr))[-1],
    )
    monkeypatch.setattr(
        mod.asf,
        "hifigan",
        lambda data, time_rebin, output: (called.append("m4"), _writer(output), (np.zeros(8), 22050))[-1],
    )
    monkeypatch.setattr(mod, "_resolve_musicnet_input_wav", lambda output_dir: str(tmp_path / "musicnet_input.wav"))
    monkeypatch.setattr(
        mod.asf,
        "musicnet",
        lambda input_audio, decoder_id, checkpoint_type, sr, output: (called.append("m5"), _writer(output), (np.zeros(8), sr))[-1],
    )

    outputs = mod.reproduce_legacy_wavs(tmp_path)

    assert called == ["m0", "m1", "m2", "m3", "m4", "m5"]
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0
