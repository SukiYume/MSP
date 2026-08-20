"""Command-line access to the single RadioSonify execution pipeline."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import click
import numpy as np

from . import __version__

_INPUT_NPY = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path)


def _parse_setting(raw: str, *, label: str) -> tuple[str, Any]:
    """Parse one repeatable ``key=value`` command-line setting."""
    key, separator, value = raw.partition("=")
    if not separator or not key.strip():
        raise click.UsageError(f"--{label} expects key=value, got: {raw!r}")
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value
    return key.strip(), parsed


def _collect(settings: tuple[str, ...], *, label: str) -> dict[str, Any]:
    """Collect repeatable settings and reject duplicate keys explicitly."""
    result: dict[str, Any] = {}
    for item in settings:
        key, value = _parse_setting(item, label=label)
        if key in result:
            raise click.UsageError(f"--{label} repeats setting {key!r}")
        result[key] = value
    return result


def _load_array(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


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
@click.version_option(version=__version__, prog_name="radiosonify")
def main() -> None:
    """RadioSonify converts profiles, matrices, and layer stacks to audio."""


def _extra_note(spec: Any) -> str:
    if spec.optional_extra is None:
        return ""
    return f" [pip install radiosonify[{spec.optional_extra}]]"


def _echo_defaults(defaults: Any, *, indent: str) -> None:
    if not defaults:
        click.echo(f"{indent}(none)")
        return
    width = max(len(name) for name in defaults)
    for name, value in defaults.items():
        click.echo(f"{indent}{name:{width}s}  default: {value!r}")


@main.command("list-methods")
def list_methods() -> None:
    """List primary methods and optional audio postprocessors."""
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
    """List every shared, method, grouped, and postprocessor setting."""
    from .preprocessing import preprocessing_defaults
    from .registry import available_methods, available_postprocessors

    click.echo("preprocess (--preprocess KEY=VALUE):")
    _echo_defaults(preprocessing_defaults(), indent="  ")
    click.echo("\nmethod parameters (--method-param KEY=VALUE):")
    for method in available_methods():
        click.echo(f"  {method.name}:")
        _echo_defaults(method.defaults, indent="    ")
        for group, group_defaults in (method.grouped_defaults or {}).items():
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
    help="Physical time span represented by the input, in seconds.",
)
@click.option("--method", default="auto", show_default=True, help="Method name or 'auto'.")
@click.option(
    "--data-type",
    default=None,
    help="Override dimensional inference: profile, matrix, or layered_matrix.",
)
@click.option("--speed", default=1.0, show_default=True, type=click.FloatRange(min=1e-9))
@click.option("--repeat", default=None, type=click.IntRange(1))
@click.option("--preserve-pitch", is_flag=True)
@click.option("--output-sr", default=None, type=click.IntRange(1))
@click.option("--time-axis", default=None, type=int)
@click.option("--layer-axis", default=None, type=int)
@click.option(
    "--preprocess",
    "preprocess_settings",
    multiple=True,
    metavar="KEY=VALUE",
    help="Shared preprocessing setting; repeat for multiple values.",
)
@click.option(
    "--method-param",
    "method_settings",
    multiple=True,
    metavar="KEY=VALUE",
    help="Selected method setting; see list-settings.",
)
@click.option("--postprocess", default=None, help="Optional audio postprocessor.")
@click.option(
    "--postprocess-param",
    "postprocess_settings",
    multiple=True,
    metavar="KEY=VALUE",
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
    """Run the unified, provenance-preserving sonification pipeline."""
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


if __name__ == "__main__":  # pragma: no cover
    main()
