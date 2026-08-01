"""Backward-compatible RadioSonify command-line interface."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

_INPUT_NPY = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path)
_INPUT_WAV = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path)


@click.group()
@click.version_option(package_name="radiosonify")
def main() -> None:
    """RadioSonify - convert radio profiles and dynamic spectra to audio."""


@main.command("list-methods")
def list_methods() -> None:
    """List available sonification methods and postprocessors."""
    from .registry import available_methods, available_postprocessors

    for method in available_methods():
        click.echo(f"  {method.name:12s}  {method.description}")
    for postprocessor in available_postprocessors():
        click.echo(f"  {postprocessor.name:12s}  {postprocessor.description} (postprocessor)")


@main.command()
@click.option("--input", "input_path", required=True, type=_INPUT_NPY, help="Input .npy file")
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--sr", default=48_000, show_default=True, type=click.IntRange(1))
@click.option("--duration", default=10.0, show_default=True, type=click.FloatRange(min=0.001))
@click.option("--repeat", default=10, show_default=True, type=click.IntRange(1))
@click.option(
    "--instrument", default="violin", show_default=True, type=click.Choice(["violin", "piano"])
)
@click.option("--no-instrument", is_flag=True, help="Disable instrument-response convolution")
@click.option("--downsample", default=None, type=click.IntRange(1))
def profile(
    input_path: Path,
    output_path: Path,
    sr: int,
    duration: float,
    repeat: int,
    instrument: str,
    no_instrument: bool,
    downsample: int | None,
) -> None:
    """Sonify a profile, or the time profile of a 2-D array."""
    from .profile import profile_to_wave

    data = np.load(input_path, allow_pickle=False)
    profile_to_wave(
        data,
        sr=sr,
        duration=duration,
        repeat=repeat,
        instrument=None if no_instrument else instrument,
        time_downsample=downsample,
        output=output_path,
    )
    click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=_INPUT_NPY, help="Input .npy file")
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--sr", default=48_000, show_default=True, type=click.IntRange(1))
@click.option("--duration", default=2.0, show_default=True, type=click.FloatRange(min=0.001))
@click.option("--freq", default=1_000.0, show_default=True, type=click.FloatRange(min=0.001))
@click.option("--repeat", default=1, show_default=True, type=click.IntRange(1))
@click.option("--compression", default=99.0, show_default=True, type=click.FloatRange(min=0.0))
@click.option("--downsample", default=None, type=click.IntRange(1))
def amplitude(
    input_path: Path,
    output_path: Path,
    sr: int,
    duration: float,
    freq: float,
    repeat: int,
    compression: float,
    downsample: int | None,
) -> None:
    """Sonify a profile by mapping amplitude to carrier loudness."""
    from .amplitude import amplitude_modulate

    data = np.load(input_path, allow_pickle=False)
    amplitude_modulate(
        data,
        sr=sr,
        duration=duration,
        freq=freq,
        repeat=repeat,
        compression=compression,
        time_downsample=downsample,
        output=output_path,
    )
    click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=_INPUT_NPY, help="Input .npy file")
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--sr", default=48_000, show_default=True, type=click.IntRange(1))
@click.option("--n-iter", default=64, show_default=True, type=click.IntRange(1))
@click.option(
    "--n-mels", default=None, type=click.IntRange(1), help="Deprecated alias for --freq-rebin"
)
@click.option("--n-fft", default=4_096, show_default=True, type=click.IntRange(2))
@click.option("--time-rebin", default=None, type=click.IntRange(1))
@click.option("--freq-rebin", default=None, type=click.IntRange(1))
@click.option("--clean", is_flag=True, help="Apply burst cleaning")
def griffinlim(
    input_path: Path,
    output_path: Path,
    sr: int,
    n_iter: int,
    n_mels: int | None,
    n_fft: int,
    time_rebin: int | None,
    freq_rebin: int | None,
    clean: bool,
) -> None:
    """Sonify a dynamic spectrum with Griffin-Lim."""
    from .griffinlim import griffinlim as run_griffinlim

    if n_mels is not None and freq_rebin is not None:
        raise click.UsageError("--n-mels and --freq-rebin cannot be supplied together")
    data = np.load(input_path, allow_pickle=False)
    run_griffinlim(
        data,
        sr=sr,
        n_iter=n_iter,
        n_fft=n_fft,
        time_rebin=time_rebin,
        freq_rebin=freq_rebin if freq_rebin is not None else n_mels,
        clean=clean,
        output=output_path,
    )
    click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=_INPUT_NPY, help="Input .npy file")
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--time-rebin", default=None, type=click.IntRange(1))
@click.option("--time-smoothing", default=None, type=click.FloatRange(min=0.0))
@click.option("--clean", is_flag=True, help="Apply burst cleaning")
def hifigan(
    input_path: Path,
    output_path: Path,
    time_rebin: int | None,
    time_smoothing: float | None,
    clean: bool,
) -> None:
    """Sonify a dynamic spectrum with the optional HiFi-GAN backend."""
    from .hifigan import hifigan as run_hifigan

    data = np.load(input_path, allow_pickle=False)
    run_hifigan(
        data,
        time_rebin=time_rebin,
        time_smoothing=time_smoothing,
        clean=clean,
        output=output_path,
    )
    click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=_INPUT_WAV, help="Input .wav file")
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--decoder-id", default=2, show_default=True, type=click.IntRange(0, 5))
@click.option(
    "--checkpoint-type",
    default="bestmodel",
    show_default=True,
    type=click.Choice(["bestmodel", "lastmodel"]),
)
@click.option("--split-size", default=20, show_default=True, type=click.IntRange(1))
@click.option("--num-threads", default=1, show_default=True, type=click.IntRange(1))
@click.option("--seed", default=0, show_default=True, type=click.IntRange(0))
def musicnet(
    input_path: Path,
    output_path: Path,
    decoder_id: int,
    checkpoint_type: str,
    split_size: int,
    num_threads: int,
    seed: int,
) -> None:
    """Apply the optional MusicNet postprocessor to a WAV file."""
    from .musicnet import musicnet as run_musicnet

    run_musicnet(
        input_path,
        decoder_id=decoder_id,
        checkpoint_type=checkpoint_type,
        split_size=split_size,
        num_threads=num_threads,
        seed=seed,
        output=output_path,
    )
    click.echo(f"Saved to {output_path}")


@main.command("download-examples")
@click.option("--dest", default="./data", show_default=True, type=click.Path(path_type=Path))
def download_examples(dest: Path) -> None:
    """Download the pinned example arrays."""
    from .hub import EXAMPLE_MAP, load_example

    dest.mkdir(parents=True, exist_ok=True)
    for name, filename in EXAMPLE_MAP.items():
        click.echo(f"Downloading {name} ({filename})...")
        np.save(dest / filename, load_example(name))
        click.echo(f"  Saved to {dest / filename}")
    click.echo("Done!")
