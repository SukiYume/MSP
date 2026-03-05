from __future__ import annotations

import argparse
import time
from pathlib import Path
import torch

import numpy as np

import radiosonify as asf
from radiosonify.hub import get_data_path

# ── terminal helpers ──────────────────────────────────────────────────────────
_W = 72  # total line width

def _banner(title: str) -> None:
    bar = "─" * _W
    print(f"\n┌{bar}┐")
    print(f"│  {title:<{_W - 2}}│")
    print(f"└{bar}┘")


def _step(n: int, total: int, label: str) -> None:
    pct = int(n / total * 100)
    filled = int(n / total * 20)
    bar = "█" * filled + "░" * (20 - filled)
    print(f"\n  [{bar}] step {n}/{total}  ({pct}%)")
    print(f"  ▶  {label}")


def _ok(msg: str, elapsed: float) -> None:
    print(f"  ✔  {msg}  ({elapsed:.1f}s)")


def _skip(msg: str) -> None:
    print(f"  ✗  SKIP  {msg}")


def _info(msg: str) -> None:
    print(f"  ·  {msg}")


LEGACY_OUTPUT_FILENAMES = {
    1: "Audio.wav",
    2: "Audio.wav",
    3: "Audio.wav",
    4: "RawBurst_Generated.wav",
    5: "MusicNet_Converted.wav",
}

LEGACY_METHOD_DIRS = {
    1: "method1_profile2wave",
    2: "method2_amp2loud",
    3: "method3_griffinlim",
    4: "method4_hifigan",
    5: "method5_musicnet",
}


# Shared RNG for reproducible synthetic fallback data.
_SYNTH_RNG = np.random.default_rng(42)

# Expected shapes of the example arrays used by the legacy methods.
_SYNTH_SHAPES: dict[str, tuple[int, ...]] = {
    "burst": (256, 1024),
    "raw_burst": (256, 1024),
    "profile": (256,),
}


def _load_example_or_synth(name: str) -> np.ndarray:
    """Load an example from HF Hub, falling back to synthetic data on failure."""
    try:
        arr = asf.load_example(name)
        _info(f"loaded '{name}' from HF Hub  shape={arr.shape}  dtype={arr.dtype}")
        return arr
    except Exception as exc:
        shape = _SYNTH_SHAPES[name]
        _info(
            f"[warn] Could not load '{name}' from HF Hub ({exc}); "
            f"using synthetic data shape={shape}"
        )
        if len(shape) == 2:
            return _SYNTH_RNG.lognormal(mean=0.0, sigma=1.0, size=shape)
        return _SYNTH_RNG.random(shape)


def build_legacy_output_paths(output_dir: str | Path) -> dict[int, Path]:
    root = Path(output_dir)
    return {
        method_id: root / LEGACY_METHOD_DIRS[method_id] / LEGACY_OUTPUT_FILENAMES[method_id]
        for method_id in range(1, 6)
    }


def _resolve_musicnet_input_wav(output_dir: str | Path) -> str:
    try:
        return get_data_path("Burst-wirfi.wav")
    except Exception:
        fallback = Path(output_dir) / "method5_musicnet" / "Burst-wirfi-fallback.wav"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        if not fallback.exists():
            burst = _load_example_or_synth("burst")
            # duration=2 matches the typical length of the original Burst-wirfi.wav.
            # Longer input increases musicnet generation time linearly (time ∝ audio_samples).
            asf.profile_to_wave(
                burst,
                sr=48000,
                duration=2,
                repeat=10,
                instrument=None,
                output=str(fallback),
            )
        return str(fallback)


def reproduce_legacy_wavs(
    output_dir: str | Path,
    include_musicnet: bool = True,
) -> dict[int, Path | None]:
    """Return a mapping of method_id -> output Path (or None if skipped)."""
    total_steps = 5
    output_paths = build_legacy_output_paths(output_dir)
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    outputs: dict[int, Path | None] = {i: None for i in range(1, 6)}
    timings: dict[int, float] = {}

    _banner("Loading input data")
    burst = _load_example_or_synth("burst")
    raw_burst = _load_example_or_synth("raw_burst")
    profile = _load_example_or_synth("profile")

    _banner("Running legacy methods")

    # ── Method 1 ─────────────────────────────────────────────────────────────
    _step(1, total_steps, "profile_to_wave  (instrument synthesis via violin MIDI)")
    _info(f"input: burst  shape={burst.shape}  sr=48000  duration=10s  repeat=10")
    _info(f"output → {output_paths[1]}")
    t0 = time.perf_counter()
    asf.profile_to_wave(
        burst,
        sr=48000,
        duration=10,
        repeat=10,
        instrument="violin",
        output=str(output_paths[1]),
    )
    elapsed = time.perf_counter() - t0
    timings[1] = elapsed
    _ok(f"wrote {output_paths[1].stat().st_size / 1024:.1f} KB", elapsed)
    outputs[1] = output_paths[1]

    # ── Method 2 ─────────────────────────────────────────────────────────────
    _step(2, total_steps, "amplitude_modulate  (AM carrier at 1 kHz)")
    _info(f"input: profile  shape={profile.shape}  sr=48000  duration=2s  freq=1000 Hz")
    _info(f"output → {output_paths[2]}")
    t0 = time.perf_counter()
    asf.amplitude_modulate(
        profile,
        sr=48000,
        duration=2,
        freq=1000,
        output=str(output_paths[2]),
    )
    elapsed = time.perf_counter() - t0
    timings[2] = elapsed
    _ok(f"wrote {output_paths[2].stat().st_size / 1024:.1f} KB", elapsed)
    outputs[2] = output_paths[2]

    # ── Method 3 ─────────────────────────────────────────────────────────────
    _step(3, total_steps, "griffinlim  (mel-spectrogram → waveform via Griffin-Lim)")
    _info(f"input: raw_burst  shape={raw_burst.shape}  n_iter=200  n_mels=512  n_fft=4096")
    _info(f"output → {output_paths[3]}")
    t0 = time.perf_counter()
    asf.griffinlim(
        raw_burst,
        sr=48000,
        n_iter=200,
        n_mels=512,
        n_fft=4096,
        time_rebin=128,
        freq_rebin=512,
        output=str(output_paths[3]),
    )
    elapsed = time.perf_counter() - t0
    timings[3] = elapsed
    _ok(f"wrote {output_paths[3].stat().st_size / 1024:.1f} KB", elapsed)
    outputs[3] = output_paths[3]

    # ── Method 4 ─────────────────────────────────────────────────────────────
    _step(4, total_steps, "hifigan  (neural vocoder – requires torch + scikit-image)")
    _info(f"input: raw_burst  shape={raw_burst.shape}  time_rebin=128")
    _info(f"output → {output_paths[4]}")
    try:
        t0 = time.perf_counter()
        asf.hifigan(
            raw_burst,
            time_rebin=128,
            output=str(output_paths[4]),
        )
        elapsed = time.perf_counter() - t0
        timings[4] = elapsed
        _ok(f"wrote {output_paths[4].stat().st_size / 1024:.1f} KB", elapsed)
        outputs[4] = output_paths[4]
    except ImportError as exc:
        _skip(f"method4 (hifigan): {exc}")
        _info("install with: pip install radiosonify[hifigan]")

    # ── Method 5 ─────────────────────────────────────────────────────────────
    _step(5, total_steps, "musicnet  (WaveNet decoder – requires torch)")
    if include_musicnet:
        try:
            musicnet_input = _resolve_musicnet_input_wav(output_dir)
            _info(f"input wav: {musicnet_input}")
            _info(f"decoder_id=2  checkpoint=bestmodel  sr=48000")
            _info(f"output → {output_paths[5]}")
            t0 = time.perf_counter()
            asf.musicnet(
                musicnet_input,
                decoder_id=2,
                checkpoint_type="bestmodel",
                sr=48000,
                output=str(output_paths[5]),
            )
            elapsed = time.perf_counter() - t0
            timings[5] = elapsed
            _ok(f"wrote {output_paths[5].stat().st_size / 1024:.1f} KB", elapsed)
            outputs[5] = output_paths[5]
        except ImportError as exc:
            _skip(f"method5 (musicnet): {exc}")
            _info("install with: pip install radiosonify[musicnet]")
    else:
        _skip("method5 (musicnet): skipped via --skip-musicnet")

    _print_summary(outputs, timings)
    return outputs


_METHOD_LABELS = {
    1: "profile_to_wave   ",
    2: "amplitude_modulate",
    3: "griffinlim        ",
    4: "hifigan           ",
    5: "musicnet          ",
}


def _print_summary(outputs: dict[int, Path | None], timings: dict[int, float]) -> None:
    bar = "─" * _W
    print(f"\n┌{bar}┐")
    print(f"│  {'Summary':<{_W - 2}}│")
    print(f"├{bar}┤")
    total = 0.0
    for mid in range(1, 6):
        path = outputs.get(mid)
        label = _METHOD_LABELS[mid]
        if path is not None:
            elapsed = timings.get(mid, 0.0)
            total += elapsed
            size_kb = path.stat().st_size / 1024
            status = f"OK   {elapsed:6.1f}s   {size_kb:8.1f} KB   {path}"
        else:
            status = "SKIP"
        print(f"│  method{mid}  {label}  {status:<{_W - 24}}│")
    print(f"├{bar}┤")
    print(f"│  total  {total:.1f}s{'':<{_W - 12 - len(f'{total:.1f}')}}│")
    print(f"└{bar}┘")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce legacy method WAV outputs using current RadioSonify APIs"
    )
    parser.add_argument(
        "--output-dir",
        default="examples/legacy_outputs",
        help="Directory to write generated WAV files",
    )
    parser.add_argument(
        "--skip-musicnet",
        action="store_true",
        help="Skip method 5 generation",
    )
    args = parser.parse_args()

    _banner("reproduce_legacy_wavs.py  –  RadioSonify example runner")
    _info(f"output dir : {Path(args.output_dir).resolve()}")
    _info(f"skip-musicnet : {args.skip_musicnet}")

    t_start = time.perf_counter()
    reproduce_legacy_wavs(args.output_dir, include_musicnet=not args.skip_musicnet)
    _info(f"wall time: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
