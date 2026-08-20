"""RadioSonify command-line interface.

``sonify`` is the unified entry point and mirrors :func:`radiosonify.sonify`:
pick a method, state how many seconds of physical time the data represents, and
control the output length with ``--speed`` and ``--repeat``. The remaining
commands are direct low-level adapters; documented deprecated aliases remain
available while data-domain controls migrate to ``--preprocess``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import click
import numpy as np

from ._perceptual_config import (
    PERCEPTUAL_CHOICES,
    PERCEPTUAL_DEFAULT_DURATION,
    PERCEPTUAL_DEFAULTS,
)

_INPUT_NPY = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path)
_INPUT_WAV = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path)


def _parse_setting(raw: str, *, label: str) -> tuple[str, Any]:
    """把 ``key=value`` 解析成 Python 值。

    值用 :func:`ast.literal_eval` 解析，因此 ``mad``、``8``、``0.75``、``None``、
    ``(5,95)`` 都能按预期得到字符串、整数、浮点、None 和元组；解析不出来的就
    按字符串处理，这样 ``--preprocess scale_statistic=mad`` 不用加引号。
    """
    key, separator, value = raw.partition("=")
    if not separator or not key.strip():
        raise click.UsageError(f"--{label} expects key=value, got: {raw!r}")
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value
    return key.strip(), parsed


def _collect(settings: tuple[str, ...], *, label: str) -> dict[str, Any]:
    return dict(_parse_setting(item, label=label) for item in settings)


def _collect_perceptual_groups(method_params: dict[str, Any]) -> dict[str, Any]:
    """Parse grouped voice and event settings shared by both perceptual commands."""
    resolved = method_params.copy()
    for name, label in (("voice_params", "voice-param"), ("event_params", "event-param")):
        settings = resolved[name]
        resolved[name] = _collect(settings, label=label) if settings else None
    return resolved


def _load_array(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _preprocessed(
    path: Path,
    settings: tuple[str, ...],
    *,
    legacy: dict[str, Any] | None = None,
) -> np.ndarray:
    """低层命令共用的前置标准化，参数与统一 API 完全一致。"""
    from .preprocessing import preprocess

    params = _collect(settings, label="preprocess")
    for name, value in (legacy or {}).items():
        if value is None:
            continue
        if name in params:
            raise click.UsageError(
                f"legacy option for {name} cannot be combined with --preprocess {name}=..."
            )
        params[name] = value
    return preprocess(_load_array(path), **params)


_preprocess_option = click.option(
    "--preprocess",
    "preprocess_settings",
    multiple=True,
    metavar="KEY=VALUE",
    help=(
        "Shared preprocessing setting, repeatable. Examples: "
        "scale_statistic=mad, clip_percentiles=(5,95), time_rebin=2048, "
        "nan_policy=propagate."
    ),
)


def _perceptual_options(command: Any) -> Any:
    """Apply compact mapping controls and two grouped extension mappings."""
    options = (
        click.option(
            "--sr",
            default=PERCEPTUAL_DEFAULTS["sr"],
            show_default=True,
            type=click.IntRange(1),
        ),
        click.option(
            "--duration",
            default=PERCEPTUAL_DEFAULT_DURATION,
            show_default=True,
            type=click.FloatRange(min=0.001),
        ),
        click.option(
            "--min-freq",
            default=PERCEPTUAL_DEFAULTS["min_freq"],
            show_default=True,
            type=click.FloatRange(min=0.001),
        ),
        click.option(
            "--max-freq",
            default=PERCEPTUAL_DEFAULTS["max_freq"],
            show_default=True,
            type=click.FloatRange(min=0.001),
        ),
        click.option(
            "--n-bands",
            default=PERCEPTUAL_DEFAULTS["n_bands"],
            show_default=True,
            type=click.IntRange(1),
            help="Simultaneous voices; omitted selects approximately one per auditory ERB.",
        ),
        click.option(
            "--value-scale",
            default=PERCEPTUAL_DEFAULTS["value_scale"],
            show_default=True,
            type=click.Choice(PERCEPTUAL_CHOICES["value_scale"]),
        ),
        click.option(
            "--gamma",
            default=PERCEPTUAL_DEFAULTS["gamma"],
            show_default=True,
            type=click.FloatRange(min=0.001),
        ),
        click.option(
            "--frequency-order",
            default=PERCEPTUAL_DEFAULTS["frequency_order"],
            show_default=True,
            type=click.Choice(PERCEPTUAL_CHOICES["frequency_order"]),
        ),
        click.option(
            "--frequency-scale",
            default=PERCEPTUAL_DEFAULTS["frequency_scale"],
            show_default=True,
            type=click.Choice(PERCEPTUAL_CHOICES["frequency_scale"]),
        ),
        click.option(
            "--timbre",
            default=PERCEPTUAL_DEFAULTS["timbre"],
            show_default=True,
            type=click.Choice(PERCEPTUAL_CHOICES["timbre"]),
        ),
        click.option(
            "--mapping-level-db",
            default=PERCEPTUAL_DEFAULTS["mapping_level_db"],
            show_default=True,
            type=click.FloatRange(max=0.0),
            help="Gain of the complete continuous mapping.",
        ),
        click.option(
            "--ambient-level-db",
            default=PERCEPTUAL_DEFAULTS["ambient_level_db"],
            show_default=True,
            type=click.FloatRange(max=0.0),
            help="Absolute-brightness ambience relative to temporal detail.",
        ),
        click.option(
            "--voice-param",
            "voice_params",
            multiple=True,
            default=(),
            metavar="KEY=VALUE",
            help="Advanced waveform setting, repeatable.",
        ),
        click.option(
            "--event-voice",
            default=PERCEPTUAL_DEFAULTS["event_voice"],
            show_default=True,
            type=click.Choice(PERCEPTUAL_CHOICES["event_voice"]),
        ),
        click.option(
            "--event-param",
            "event_params",
            multiple=True,
            default=(),
            metavar="KEY=VALUE",
            help="Optional event-decorator setting, repeatable.",
        ),
        click.option(
            "--attack-ms",
            default=PERCEPTUAL_DEFAULTS["attack_ms"],
            show_default=True,
            type=click.FloatRange(min=0),
        ),
        click.option(
            "--release-ms",
            default=PERCEPTUAL_DEFAULTS["release_ms"],
            show_default=True,
            type=click.FloatRange(min=0),
        ),
        click.option(
            "--loudness-compensation-db",
            default=PERCEPTUAL_DEFAULTS["loudness_compensation_db"],
            show_default=True,
            type=click.FloatRange(min=0),
        ),
        click.option(
            "--rms-limit-dbfs",
            default=PERCEPTUAL_DEFAULTS["rms_limit_dbfs"],
            show_default=True,
            type=click.FloatRange(max=0),
        ),
        click.option(
            "--peak-limit-dbfs",
            default=PERCEPTUAL_DEFAULTS["peak_limit_dbfs"],
            show_default=True,
            type=click.FloatRange(max=0),
        ),
    )
    for option in reversed(options):
        command = option(command)
    return command


class _UserFacingGroup(click.Group):
    """Convert expected library failures into concise command-line errors."""

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except (click.Abort, click.exceptions.Exit, click.ClickException):
            raise
        except (ValueError, ImportError, OSError, RuntimeError) as exc:
            raise click.ClickException(str(exc)) from exc


@click.group(cls=_UserFacingGroup)
@click.version_option(package_name="radiosonify")
def main() -> None:
    """RadioSonify - convert numerical profiles, matrices, and layer stacks to audio."""


def _extra_note(spec: Any) -> str:
    """Name the optional install extra a registered entry depends on."""
    if spec.optional_extra is None:
        return ""
    return f" [pip install radiosonify[{spec.optional_extra}]]"


def _echo_defaults(defaults: Any, *, indent: str) -> None:
    """Print one settings mapping as ``name  default: value`` lines."""
    if not defaults:
        click.echo(f"{indent}(none)")
        return
    width = max(len(name) for name in defaults)
    for name, value in defaults.items():
        click.echo(f"{indent}{name:{width}s}  default: {value!r}")


@main.command("list-methods")
def list_methods() -> None:
    """List available sonification methods and postprocessors."""
    from .registry import available_methods, available_postprocessors

    for method in available_methods():
        click.echo(f"  {method.name:12s}  {method.description}{_extra_note(method)}")
    for postprocessor in available_postprocessors():
        click.echo(
            f"  {postprocessor.name:12s}  {postprocessor.description} "
            f"(postprocessor){_extra_note(postprocessor)}"
        )


@main.command("list-settings")
def list_settings() -> None:
    """List the shared preprocessing settings and each method's parameters."""
    from .preprocessing import preprocessing_defaults
    from .registry import available_methods, available_postprocessors

    click.echo("preprocess (--preprocess KEY=VALUE, shared by every method):")
    _echo_defaults(preprocessing_defaults(), indent="  ")
    click.echo("\nmethod parameters (--method-param KEY=VALUE):")
    for method in available_methods():
        click.echo(f"  {method.name}:")
        _echo_defaults(method.defaults, indent="    ")
        for group, group_defaults in (method.grouped_defaults or {}).items():
            # A grouped mapping shows up above as a bare ``None``. Expanding it
            # here is what makes its keys reachable without reading the source.
            click.echo(f"    {group} accepts:")
            _echo_defaults(group_defaults, indent="      ")
    click.echo("\npostprocess parameters (--postprocess-param KEY=VALUE):")
    for postprocessor in available_postprocessors():
        click.echo(f"  {postprocessor.name}:")
        _echo_defaults(postprocessor.defaults, indent="    ")


@main.command("sonify")
@click.option("--input", "input_path", required=True, type=_INPUT_NPY, help="Input .npy file")
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option(
    "--duration",
    "data_duration",
    required=True,
    type=click.FloatRange(min=1e-9),
    help="Physical time span the data represents, in seconds.",
)
@click.option("--method", default="auto", show_default=True, help="Method name or 'auto'.")
@click.option(
    "--data-type",
    default=None,
    help="Override the type inferred from dimensionality (profile/matrix/layered_matrix).",
)
@click.option(
    "--speed",
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=1e-9),
    help="Playback speed. 2 halves the output duration, 0.5 doubles it.",
)
@click.option(
    "--repeat",
    default=None,
    type=click.IntRange(1),
    help="Repeat the data this many times. Defaults to the method's registered value.",
)
@click.option("--preserve-pitch", is_flag=True, help="Time-stretch without shifting pitch.")
@click.option("--output-sr", default=None, type=click.IntRange(1), help="Final container rate.")
@click.option("--time-axis", default=None, type=int, help="Which input axis carries time.")
@click.option("--layer-axis", default=None, type=int, help="Which 3-D input axis holds layers.")
@_preprocess_option
@click.option(
    "--method-param",
    "method_settings",
    multiple=True,
    metavar="KEY=VALUE",
    help="Method parameter, repeatable. See 'radiosonify list-settings'.",
)
@click.option("--postprocess", default=None, help="Optional style-transfer postprocessor.")
@click.option(
    "--postprocess-param",
    "postprocess_settings",
    multiple=True,
    metavar="KEY=VALUE",
    help="Postprocessor parameter, repeatable.",
)
def sonify_command(
    input_path: Path,
    output_path: Path,
    data_duration: float,
    method: str,
    data_type: str | None,
    speed: float,
    repeat: int | None,
    preserve_pitch: bool,
    output_sr: int | None,
    time_axis: int | None,
    layer_axis: int | None,
    preprocess_settings: tuple[str, ...],
    method_settings: tuple[str, ...],
    postprocess: str | None,
    postprocess_settings: tuple[str, ...],
) -> None:
    """Sonify a 1-D, 2-D, or 3-D array through the unified pipeline."""
    from .api import sonify
    from .inputs import SonificationInput

    source = SonificationInput(
        _load_array(input_path),
        duration=data_duration,
        data_type=data_type,
        name=input_path.stem,
        time_axis=time_axis,
        layer_axis=layer_axis,
    )
    result = sonify(
        source,
        method=method,
        speed=speed,
        repeat=repeat,
        preserve_pitch=preserve_pitch,
        output_sr=output_sr,
        preprocess_params=_collect(preprocess_settings, label="preprocess"),
        method_params=_collect(method_settings, label="method-param"),
        postprocess=postprocess,
        postprocess_params=(
            _collect(postprocess_settings, label="postprocess-param")
            if postprocess_settings
            else None
        ),
        output=output_path,
    )
    click.echo(
        f"Saved to {output_path} "
        f"({result.output_duration:.3f} s at {result.sample_rate} Hz, "
        f"method={result.method}, repeat={result.repeat}, speed={result.speed:g})"
    )


@main.command()
@click.option("--input", "input_path", required=True, type=_INPUT_NPY, help="Input .npy file")
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--sr", default=48_000, show_default=True, type=click.IntRange(1))
@click.option("--duration", default=10.0, show_default=True, type=click.FloatRange(min=0.001))
@click.option(
    "--instrument", default="violin", show_default=True, type=click.Choice(["violin", "piano"])
)
@click.option("--no-instrument", is_flag=True, help="Disable instrument-response convolution")
@_preprocess_option
def profile(
    input_path: Path,
    output_path: Path,
    sr: int,
    duration: float,
    instrument: str,
    no_instrument: bool,
    preprocess_settings: tuple[str, ...],
) -> None:
    """Sonify a profile, or the time profile of a 2-D array."""
    from .profile import profile_to_wave

    profile_to_wave(
        _preprocessed(input_path, preprocess_settings),
        sr=sr,
        duration=duration,
        instrument=None if no_instrument else instrument,
        output=output_path,
    )
    click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=_INPUT_NPY, help="Input .npy file")
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--sr", default=48_000, show_default=True, type=click.IntRange(1))
@click.option("--duration", default=2.0, show_default=True, type=click.FloatRange(min=0.001))
@click.option("--freq", default=1_000.0, show_default=True, type=click.FloatRange(min=0.001))
@click.option("--compression", default=0.0, show_default=True, type=click.FloatRange(min=0.0))
@click.option("--harmonics", default=4, show_default=True, type=click.IntRange(1))
@click.option("--harmonic-decay", default=1.0, show_default=True, type=click.FloatRange(min=0.0))
@_preprocess_option
def amplitude(
    input_path: Path,
    output_path: Path,
    sr: int,
    duration: float,
    freq: float,
    compression: float,
    harmonics: int,
    harmonic_decay: float,
    preprocess_settings: tuple[str, ...],
) -> None:
    """Sonify a profile by mapping amplitude to carrier loudness."""
    from .amplitude import amplitude_modulate

    amplitude_modulate(
        _preprocessed(input_path, preprocess_settings),
        sr=sr,
        duration=duration,
        freq=freq,
        compression=compression,
        harmonics=harmonics,
        harmonic_decay=harmonic_decay,
        output=output_path,
    )
    click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=_INPUT_NPY, help="Input 2-D .npy file")
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@_perceptual_options
@_preprocess_option
def erb(
    input_path: Path,
    output_path: Path,
    preprocess_settings: tuple[str, ...],
    **method_params: Any,
) -> None:
    """Sonify a 2-D matrix with a continuous perceptual scan."""
    from .erb import erb_sonify

    erb_sonify(
        _preprocessed(input_path, preprocess_settings),
        output=output_path,
        **_collect_perceptual_groups(method_params),
    )
    click.echo(f"Saved to {output_path}")


@main.command("spatial-erb")
@click.option("--input", "input_path", required=True, type=_INPUT_NPY, help="Input 3-D .npy file")
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@_perceptual_options
@click.option("--pan", "pan_positions", multiple=True, type=click.FloatRange(-1.0, 1.0))
@click.option("--layer-gain", "layer_gains", multiple=True, type=click.FloatRange(min=0.0))
@_preprocess_option
def spatial_erb(
    input_path: Path,
    output_path: Path,
    pan_positions: tuple[float, ...],
    layer_gains: tuple[float, ...],
    preprocess_settings: tuple[str, ...],
    **method_params: Any,
) -> None:
    """Sonify a 3-D (layer, time, feature) stack across a stereo field."""
    from .spatial import spatial_sonify

    spatial_sonify(
        _preprocessed(input_path, preprocess_settings),
        pan_positions=pan_positions or None,
        layer_gains=layer_gains or None,
        output=output_path,
        **_collect_perceptual_groups(method_params),
    )
    click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=_INPUT_NPY, help="Input .npy file")
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--sr", default=48_000, show_default=True, type=click.IntRange(1))
@click.option("--n-iter", default=64, show_default=True, type=click.IntRange(1))
@click.option(
    "--n-mels",
    default=None,
    type=click.IntRange(1),
    help="Deprecated alias for --freq-rebin.",
)
@click.option("--n-fft", default=4_096, show_default=True, type=click.IntRange(2))
@click.option(
    "--frame-length",
    default=0.04,
    show_default=True,
    type=click.FloatRange(min=1e-9),
    help="Analysis frame length in seconds; with --sr it fixes the hop length.",
)
@click.option(
    "--preemphasis",
    default=0.0,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0, max_open=True),
    help="De-emphasis coefficient used for deliberate tonal coloring.",
)
@click.option("--max-db", default=100.0, show_default=True, type=click.FloatRange(min=1e-9))
@click.option("--ref-db", default=20.0, show_default=True, type=float)
@click.option("--time-rebin", default=None, type=click.IntRange(1), hidden=True)
@click.option("--freq-rebin", default=None, type=click.IntRange(1), hidden=True)
@_preprocess_option
def griffinlim(
    input_path: Path,
    output_path: Path,
    sr: int,
    n_iter: int,
    n_mels: int | None,
    n_fft: int,
    frame_length: float,
    preemphasis: float,
    max_db: float,
    ref_db: float,
    time_rebin: int | None,
    freq_rebin: int | None,
    preprocess_settings: tuple[str, ...],
) -> None:
    """Sonify a dynamic spectrum with Griffin-Lim."""
    from .griffinlim import griffinlim as run_griffinlim

    if n_mels is not None and freq_rebin is not None:
        raise click.UsageError("--n-mels and --freq-rebin cannot be supplied together")
    run_griffinlim(
        _preprocessed(
            input_path,
            preprocess_settings,
            legacy={
                "time_rebin": time_rebin,
                "feature_rebin": freq_rebin if freq_rebin is not None else n_mels,
            },
        ),
        sr=sr,
        n_iter=n_iter,
        n_fft=n_fft,
        frame_length=frame_length,
        preemphasis=preemphasis,
        max_db=max_db,
        ref_db=ref_db,
        output=output_path,
    )
    click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=_INPUT_NPY, help="Input .npy file")
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@_preprocess_option
def hifigan(
    input_path: Path,
    output_path: Path,
    preprocess_settings: tuple[str, ...],
) -> None:
    """Sonify a dynamic spectrum with the optional HiFi-GAN backend."""
    from .hifigan import hifigan as run_hifigan

    run_hifigan(
        _preprocessed(input_path, preprocess_settings),
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


@main.command()
@click.option("--input", "input_path", required=True, type=_INPUT_WAV, help="Input .wav file")
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
    help="Trusted exported RAVE TorchScript model (.ts)",
)
@click.option("--device", default="auto", show_default=True, type=str)
def rave(input_path: Path, output_path: Path, model_path: Path, device: str) -> None:
    """Apply a user-supplied exported RAVE model to a WAV file."""
    from .rave import rave as run_rave

    run_rave(
        input_path,
        model_path=model_path,
        device=device,
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


if __name__ == "__main__":  # pragma: no cover - exercised through the console entry point
    main()
