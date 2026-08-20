from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner

from radiosonify._perceptual_config import (
    PERCEPTUAL_DEFAULT_DURATION,
    PERCEPTUAL_DEFAULTS,
)
from radiosonify.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.parametrize("command_name", ["erb", "spatial-erb"])
def test_perceptual_low_level_cli_defaults_share_the_engine_source(command_name):
    command = main.commands[command_name]
    defaults = {parameter.name: parameter.default for parameter in command.params}

    assert defaults["duration"] == PERCEPTUAL_DEFAULT_DURATION
    scalar_defaults = {
        name: value
        for name, value in PERCEPTUAL_DEFAULTS.items()
        if name not in {"voice_params", "event_params"}
    }
    assert {name: defaults[name] for name in scalar_defaults} == scalar_defaults
    assert defaults["voice_params"] == ()
    assert defaults["event_params"] == ()


@pytest.fixture
def profile_file(tmp_path):
    path = tmp_path / "profile.npy"
    np.save(path, np.random.default_rng(42).random(32))
    return path


@pytest.fixture
def spectrum_file(tmp_path):
    path = tmp_path / "spectrum.npy"
    np.save(path, np.random.default_rng(42).random((16, 16)))
    return path


@pytest.fixture
def layered_file(tmp_path):
    path = tmp_path / "layers.npy"
    np.save(path, np.random.default_rng(42).random((4, 8, 8)))
    return path


def test_help_and_version(runner):
    help_result = runner.invoke(main, ["--help"])
    version_result = runner.invoke(main, ["--version"])

    assert help_result.exit_code == 0
    assert "RadioSonify" in help_result.output
    assert version_result.exit_code == 0
    assert "0.2.0" in version_result.output


def test_list_methods(runner):
    result = runner.invoke(main, ["list-methods"])

    assert result.exit_code == 0
    assert "profile" in result.output
    assert "griffinlim" in result.output
    assert "musicnet" in result.output
    assert "erb" in result.output
    assert "spatial_erb" in result.output
    assert "rave" in result.output


def test_profile_and_amplitude_legacy_commands(runner, profile_file, tmp_path):
    profile_output = tmp_path / "profile.wav"
    amplitude_output = tmp_path / "amplitude.wav"

    profile_result = runner.invoke(
        main,
        [
            "profile",
            "--input",
            str(profile_file),
            "--output",
            str(profile_output),
            "--duration",
            "0.01",
            "--no-instrument",
        ],
    )
    amplitude_result = runner.invoke(
        main,
        [
            "amplitude",
            "--input",
            str(profile_file),
            "--output",
            str(amplitude_output),
            "--duration",
            "0.01",
        ],
    )

    assert profile_result.exit_code == 0, profile_result.output
    assert amplitude_result.exit_code == 0, amplitude_result.output
    assert profile_output.is_file()
    assert amplitude_output.is_file()


def test_griffinlim_rebinning_moved_to_shared_preprocessing(runner, spectrum_file, tmp_path):
    """新脚本通过统一的 --preprocess 表达重分箱。"""
    output = tmp_path / "griffinlim.wav"
    result = runner.invoke(
        main,
        [
            "griffinlim",
            "--input",
            str(spectrum_file),
            "--output",
            str(output),
            "--sr",
            "800",
            "--n-iter",
            "1",
            "--n-fft",
            "32",
            "--preprocess",
            "time_rebin=8",
            "--preprocess",
            "feature_rebin=16",
        ],
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert output.is_file()


def test_griffinlim_legacy_n_mels_alias_routes_to_shared_preprocessing(
    runner, spectrum_file, tmp_path
):
    output = tmp_path / "legacy-griffinlim.wav"
    result = runner.invoke(
        main,
        [
            "griffinlim",
            "--input",
            str(spectrum_file),
            "--output",
            str(output),
            "--sr",
            "800",
            "--n-iter",
            "1",
            "--n-fft",
            "32",
            "--n-mels",
            "16",
            "--time-rebin",
            "8",
        ],
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert output.is_file()


def test_expected_library_error_is_reported_without_a_traceback(runner, spectrum_file, tmp_path):
    result = runner.invoke(
        main,
        [
            "sonify",
            "--input",
            str(spectrum_file),
            "--output",
            str(tmp_path / "unknown.wav"),
            "--duration",
            "1",
            "--method",
            "unknown-method",
        ],
    )

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "unknown method" in result.output
    assert "Traceback" not in result.output


def test_erb_and_spatial_erb_commands(runner, spectrum_file, layered_file, tmp_path):
    matrix_output = tmp_path / "erb.wav"
    layered_output = tmp_path / "spatial.wav"

    matrix = runner.invoke(
        main,
        [
            "erb",
            "--input",
            str(spectrum_file),
            "--output",
            str(matrix_output),
            "--sr",
            "8000",
            "--max-freq",
            "3000",
            "--duration",
            "0.01",
            "--n-bands",
            "4",
            "--timbre",
            "sine",
        ],
    )
    layered = runner.invoke(
        main,
        [
            "spatial-erb",
            "--input",
            str(layered_file),
            "--output",
            str(layered_output),
            "--sr",
            "8000",
            "--max-freq",
            "3000",
            "--duration",
            "0.01",
            "--n-bands",
            "4",
            "--timbre",
            "retro_digital",
        ],
    )

    assert matrix.exit_code == 0, f"{matrix.output}\n{matrix.exception}"
    assert layered.exit_code == 0, f"{layered.output}\n{layered.exception}"
    assert matrix_output.is_file()
    assert layered_output.is_file()


def test_unknown_preprocess_setting_is_rejected(runner, spectrum_file, tmp_path):
    result = runner.invoke(
        main,
        [
            "griffinlim",
            "--input",
            str(spectrum_file),
            "--output",
            str(tmp_path / "out.wav"),
            "--preprocess",
            "freq_rebin=8",
        ],
    )

    assert result.exit_code != 0


def test_malformed_preprocess_setting_is_rejected(runner, spectrum_file, tmp_path):
    result = runner.invoke(
        main,
        [
            "griffinlim",
            "--input",
            str(spectrum_file),
            "--output",
            str(tmp_path / "out.wav"),
            "--preprocess",
            "scale_statistic",
        ],
    )

    assert result.exit_code != 0
    assert "key=value" in result.output


@pytest.mark.parametrize(
    ("fixture_name", "extra"),
    [
        ("profile_file", []),
        ("spectrum_file", ["--method", "erb"]),
        ("layered_file", []),
    ],
)
def test_unified_sonify_command_covers_every_dimensionality(
    runner, request, tmp_path, fixture_name, extra
):
    """统一入口在 CLI 上和 Python API 一样：选方法、给数据时长、调 speed/repeat。"""
    source = request.getfixturevalue(fixture_name)
    output = tmp_path / f"{fixture_name}.wav"

    result = runner.invoke(
        main,
        [
            "sonify",
            "--input",
            str(source),
            "--output",
            str(output),
            "--duration",
            "0.4",
            "--speed",
            "2",
            "--repeat",
            "2",
            "--preprocess",
            "scale_statistic=mad",
            "--preprocess",
            "clip_percentiles=(5,95)",
            *extra,
        ],
    )

    assert result.exit_code == 0, f"{result.output} {result.exception}"
    assert output.is_file()
    # duration * repeat / speed = 0.4 * 2 / 2
    assert "0.400 s" in result.output


def test_unified_sonify_command_accepts_axis_declarations(runner, tmp_path):
    source = tmp_path / "channels_last.npy"
    np.save(source, np.random.default_rng(1).random((8, 6, 3)))
    output = tmp_path / "axes.wav"

    result = runner.invoke(
        main,
        [
            "sonify",
            "--input",
            str(source),
            "--output",
            str(output),
            "--duration",
            "0.2",
            "--layer-axis",
            "2",
        ],
    )

    assert result.exit_code == 0, f"{result.output} {result.exception}"
    assert output.is_file()


def test_list_settings_documents_the_shared_preprocessing_surface(runner):
    result = runner.invoke(main, ["list-settings"])

    assert result.exit_code == 0
    assert "scale_statistic" in result.output
    assert "nan_policy" in result.output
    assert "normalization_scope" in result.output


def test_input_existence_and_decoder_range_validation(runner, tmp_path):
    missing = runner.invoke(
        main,
        ["profile", "--input", str(tmp_path / "missing.npy"), "--output", "out.wav"],
    )
    decoder = runner.invoke(
        main,
        [
            "musicnet",
            "--input",
            str(tmp_path / "missing.wav"),
            "--output",
            "out.wav",
            "--decoder-id",
            "6",
        ],
    )

    assert missing.exit_code != 0
    assert decoder.exit_code != 0


def test_list_methods_names_the_optional_extra_each_backend_needs(runner):
    result = runner.invoke(main, ["list-methods"])

    assert result.exit_code == 0
    assert "pip install radiosonify[hifigan]" in result.output
    assert "pip install radiosonify[musicnet]" in result.output
    assert "pip install radiosonify[rave]" in result.output
    # Dependency-free methods stay unannotated, so the marker means "extra required".
    erb_line = next(line for line in result.output.splitlines() if line.strip().startswith("erb "))
    assert "pip install" not in erb_line


def test_list_settings_prints_every_registered_default(runner):
    from radiosonify.registry import available_methods, available_postprocessors

    result = runner.invoke(main, ["list-settings"])

    assert result.exit_code == 0
    # A registered default the user never sees is a default the user cannot
    # reproduce; every method and postprocessor value is printed. This is how
    # the unified API's `instrument: None` becomes visible next to the
    # low-level `profile_to_wave(instrument="violin")` compatibility default.
    assert "instrument" in result.output
    assert "default: None" in result.output
    for spec in (*available_methods(), *available_postprocessors()):
        for name, value in spec.defaults.items():
            assert f"{name}" in result.output
            assert f"default: {value!r}" in result.output
    assert "(none)" in result.output  # hifigan registers no method parameters


def test_griffinlim_command_exposes_every_registered_method_parameter(
    runner, spectrum_file, tmp_path
):
    from radiosonify.registry import resolve_method

    command = main.commands["griffinlim"]
    options = {parameter.name for parameter in command.params}
    assert set(resolve_method("griffinlim", "matrix").parameters) <= options

    output = tmp_path / "gl.wav"
    result = runner.invoke(
        main,
        [
            "griffinlim",
            "--input",
            str(spectrum_file),
            "--output",
            str(output),
            "--sr",
            "8000",
            "--n-iter",
            "2",
            "--n-fft",
            "512",
            "--frame-length",
            "0.02",
            "--preemphasis",
            "0.5",
            "--max-db",
            "80",
            "--ref-db",
            "10",
        ],
    )

    assert result.exit_code == 0, f"{result.output} {result.exception}"
    assert output.is_file()


def test_list_settings_expands_the_grouped_perceptual_mappings(runner):
    from radiosonify._perceptual_config import EVENT_DEFAULTS, VOICE_DEFAULTS

    result = runner.invoke(main, ["list-settings"])

    assert result.exit_code == 0
    assert "voice_params accepts:" in result.output
    assert "event_params accepts:" in result.output
    # The grouped mappings default to None, so their keys are reachable only
    # when the discovery command expands them.
    for name, value in (*VOICE_DEFAULTS.items(), *EVENT_DEFAULTS.items()):
        assert f"{name}" in result.output
        assert f"default: {value!r}" in result.output
