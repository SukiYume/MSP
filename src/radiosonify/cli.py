"""RadioSonify command-line interface."""

from __future__ import annotations

import click
import numpy as np


@click.group()
@click.version_option()
def main():
    """RadioSonify - Convert radio telescope data into audible sound."""
    pass


@main.command()
def list_methods():
    """List available sonification methods."""
    methods = [
        ("profile", "Convert pulse profile to waveform with instrument convolution"),
        ("amplitude", "Amplitude-modulated sine wave"),
        ("griffinlim", "Griffin-Lim phase reconstruction vocoder"),
        ("hifigan", "HiFi-GAN neural vocoder (requires torch)"),
        ("musicnet", "WaveNet music style transfer (requires torch; CUDA recommended)"),
    ]
    for name, desc in methods:
        click.echo(f"  {name:12s}  {desc}")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="Input .npy file")
@click.option("--output", "output_path", required=True, help="Output .wav file")
@click.option("--sr", default=48000, type=click.IntRange(1), help="Sample rate (Hz)")
@click.option("--duration", default=10.0, type=click.FloatRange(min=0.001), help="Duration (seconds)")
@click.option("--repeat", default=10, type=click.IntRange(1), help="Profile repeat count")
@click.option("--instrument", default="violin", help="Instrument (violin/piano)")
@click.option("--no-instrument", is_flag=True, help="Disable instrument convolution")
@click.option("--downsample", default=None, type=click.IntRange(1), help="Time downsample factor")
def profile(input_path, output_path, sr, duration, repeat, instrument, no_instrument, downsample):
    """Sonify using pulse profile to waveform (Method 1)."""
    from .profile import profile_to_wave
    data = np.load(input_path)
    inst = None if no_instrument else instrument
    profile_to_wave(data, sr=sr, duration=duration, repeat=repeat,
                    instrument=inst, time_downsample=downsample, output=output_path)
    click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="Input .npy file")
@click.option("--output", "output_path", required=True, help="Output .wav file")
@click.option("--sr", default=48000, type=click.IntRange(1), help="Sample rate (Hz)")
@click.option("--duration", default=2.0, type=click.FloatRange(min=0.001), help="Duration (seconds)")
@click.option("--freq", default=1000.0, type=click.FloatRange(min=1e-3), help="Carrier frequency (Hz)")
@click.option("--downsample", default=None, type=click.IntRange(1), help="Time downsample factor")
def amplitude(input_path, output_path, sr, duration, freq, downsample):
    """Sonify using amplitude modulation (Method 2)."""
    from .amplitude import amplitude_modulate
    data = np.load(input_path)
    amplitude_modulate(data, sr=sr, duration=duration, freq=freq,
                       time_downsample=downsample, output=output_path)
    click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="Input .npy file")
@click.option("--output", "output_path", required=True, help="Output .wav file")
@click.option("--sr", default=48000, type=click.IntRange(1), help="Sample rate (Hz)")
@click.option("--n-iter", default=200, type=click.IntRange(1), help="Griffin-Lim iterations")
@click.option("--n-mels", default=512, type=click.IntRange(1), help="Number of mel bands")
@click.option("--n-fft", default=4096, type=click.IntRange(2), help="FFT size")
@click.option("--time-rebin", default=None, type=click.IntRange(1), help="Time downsample bins")
@click.option("--freq-rebin", default=None, type=click.IntRange(1), help="Freq downsample bins")
@click.option("--clean", is_flag=True, help="Apply burst cleaning")
def griffinlim(input_path, output_path, sr, n_iter, n_mels, n_fft, time_rebin, freq_rebin, clean):
    """Sonify using Griffin-Lim vocoder (Method 3)."""
    from .griffinlim import griffinlim as gl
    data = np.load(input_path)
    gl(data, sr=sr, n_iter=n_iter, n_mels=n_mels, n_fft=n_fft,
       time_rebin=time_rebin, freq_rebin=freq_rebin, clean=clean, output=output_path)
    click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="Input .npy file")
@click.option("--output", "output_path", required=True, help="Output .wav file")
@click.option("--time-rebin", default=None, type=click.IntRange(1), help="Time downsample bins")
@click.option("--clean", is_flag=True, help="Apply burst cleaning")
def hifigan(input_path, output_path, time_rebin, clean):
    """Sonify using HiFi-GAN neural vocoder (Method 4)."""
    from .hifigan import hifigan as hf
    data = np.load(input_path)
    hf(data, time_rebin=time_rebin, clean=clean, output=output_path)
    click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="Input .wav file")
@click.option("--output", "output_path", required=True, help="Output .wav file")
@click.option("--decoder-id", default=2, type=click.IntRange(0, 5), help="Style decoder (0-5)")
@click.option("--checkpoint-type", default="bestmodel", help="bestmodel or lastmodel")
@click.option("--sr", default=48000, type=click.IntRange(1), help="Sample rate (Hz)")
def musicnet(input_path, output_path, decoder_id, checkpoint_type, sr):
    """Sonify using WaveNet style transfer (Method 5)."""
    from .musicnet import musicnet as mn
    mn(input_path, decoder_id=decoder_id, checkpoint_type=checkpoint_type,
       sr=sr, output=output_path)
    click.echo(f"Saved to {output_path}")


@main.command("download-examples")
@click.option("--dest", default="./data", help="Destination directory")
def download_examples(dest):
    """Download example data files from Hugging Face Hub."""
    import os
    from .hub import load_example, EXAMPLE_MAP
    os.makedirs(dest, exist_ok=True)
    for name, filename in EXAMPLE_MAP.items():
        click.echo(f"Downloading {name} ({filename})...")
        data = load_example(name)
        out_path = os.path.join(dest, filename)
        np.save(out_path, data)
        click.echo(f"  Saved to {out_path}")
    click.echo("Done!")
