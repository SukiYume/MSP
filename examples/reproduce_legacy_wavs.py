from __future__ import annotations

import argparse
from pathlib import Path

import astrosonify as asf
from astrosonify.hub import get_data_path


LEGACY_OUTPUT_FILENAMES = {
    0: "Audio.wav",
    1: "Audio.wav",
    2: "Audio.wav",
    3: "Audio.wav",
    4: "RawBurst_Generated.wav",
    5: "MusicNet_Converted.wav",
}

LEGACY_METHOD_DIRS = {
    0: "method0_astronify",
    1: "method1_profile2wave",
    2: "method2_amp2loud",
    3: "method3_griffinlim",
    4: "method4_hifigan",
    5: "method5_musicnet",
}


def build_legacy_output_paths(output_dir: str | Path) -> dict[int, Path]:
    root = Path(output_dir)
    return {
        method_id: root / LEGACY_METHOD_DIRS[method_id] / LEGACY_OUTPUT_FILENAMES[method_id]
        for method_id in range(6)
    }


def _resolve_musicnet_input_wav(output_dir: str | Path) -> str:
    try:
        return get_data_path("Burst-wirfi.wav")
    except Exception:
        fallback = Path(output_dir) / "method5_musicnet" / "Burst-wirfi-fallback.wav"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        burst = asf.load_example("burst")
        asf.profile_to_wave(
            burst,
            sr=48000,
            duration=6,
            repeat=10,
            instrument=None,
            output=str(fallback),
        )
        return str(fallback)


def reproduce_legacy_wavs(output_dir: str | Path, include_musicnet: bool = True) -> dict[int, Path]:
    outputs = build_legacy_output_paths(output_dir)
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    burst = asf.load_example("burst")
    raw_burst = asf.load_example("raw_burst")
    profile = asf.load_example("profile")

    asf.astronify_sonify(
        burst,
        note_spacing=0.01,
        time_downsample=10,
        output=str(outputs[0]),
    )

    asf.profile_to_wave(
        burst,
        sr=48000,
        duration=10,
        repeat=10,
        instrument="violin",
        output=str(outputs[1]),
    )

    asf.amplitude_modulate(
        profile,
        sr=48000,
        duration=2,
        freq=1000,
        output=str(outputs[2]),
    )

    asf.griffinlim(
        raw_burst,
        sr=48000,
        n_iter=200,
        n_mels=512,
        n_fft=4096,
        time_rebin=128,
        freq_rebin=512,
        output=str(outputs[3]),
    )

    asf.hifigan(
        raw_burst,
        time_rebin=128,
        output=str(outputs[4]),
    )

    if include_musicnet:
        musicnet_input = _resolve_musicnet_input_wav(output_dir)
        asf.musicnet(
            musicnet_input,
            decoder_id=2,
            checkpoint_type="bestmodel",
            sr=48000,
            output=str(outputs[5]),
        )

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce legacy method WAV outputs using current AstroSonify APIs")
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

    outputs = reproduce_legacy_wavs(args.output_dir, include_musicnet=not args.skip_musicnet)
    for method_id in range(6):
        if method_id == 5 and args.skip_musicnet:
            continue
        print(f"method{method_id}: {outputs[method_id]}")


if __name__ == "__main__":
    main()
