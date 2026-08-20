from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner

from radiosonify.cli import main


@pytest.fixture
def runner():
    return CliRunner()


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


def test_cli_has_one_execution_command():
    assert set(main.commands) == {
        "sonify",
        "list-methods",
        "list-settings",
        "download-examples",
    }


def test_help_and_version(runner):
    help_result = runner.invoke(main, ["--help"])
    version_result = runner.invoke(main, ["--version"])

    assert help_result.exit_code == 0
    assert "RadioSonify" in help_result.output
    assert "sonify" in help_result.output
    assert version_result.exit_code == 0
    assert "0.3.0" in version_result.output


def test_removed_low_level_commands_are_absent(runner):
    for command in ("profile", "amplitude", "erb", "spatial-erb", "griffinlim", "hifigan"):
        result = runner.invoke(main, [command])
        assert result.exit_code != 0
        assert "No such command" in result.output


def test_list_methods(runner):
    result = runner.invoke(main, ["list-methods"])

    assert result.exit_code == 0
    for name in ("profile", "amplitude", "griffinlim", "hifigan", "erb", "spatial_erb"):
        assert name in result.output
    for name in ("musicnet", "rave"):
        assert name in result.output
        assert "postprocessor" in result.output


@pytest.mark.parametrize("method", ["profile", "amplitude"])
def test_profile_methods_run_through_unified_command(runner, profile_file, tmp_path, method):
    output = tmp_path / f"{method}.wav"
    result = runner.invoke(
        main,
        [
            "sonify",
            "--input",
            str(profile_file),
            "--output",
            str(output),
            "--duration",
            "0.01",
            "--repeat",
            "1",
            "--method",
            method,
            "--method-param",
            "sr=8000",
        ],
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert output.is_file()


def test_griffinlim_uses_shared_preprocessing_and_generic_method_settings(
    runner, spectrum_file, tmp_path
):
    output = tmp_path / "griffinlim.wav"
    result = runner.invoke(
        main,
        [
            "sonify",
            "--input",
            str(spectrum_file),
            "--output",
            str(output),
            "--duration",
            "0.04",
            "--method",
            "griffinlim",
            "--preprocess",
            "time_rebin=10",
            "--preprocess",
            "feature_rebin=16",
            "--method-param",
            "sr=800",
            "--method-param",
            "n_iter=1",
            "--method-param",
            "n_fft=32",
            "--method-param",
            "frame_length=0.02",
        ],
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert output.is_file()


@pytest.mark.parametrize(
    ("fixture_name", "method", "method_settings"),
    [
        ("spectrum_file", "erb", ["sr=8000", "max_freq=3000", "n_bands=4"]),
        ("layered_file", "spatial_erb", ["sr=8000", "max_freq=3000", "n_bands=4"]),
    ],
)
def test_perceptual_methods_run_through_unified_command(
    runner, request, tmp_path, fixture_name, method, method_settings
):
    source = request.getfixturevalue(fixture_name)
    output = tmp_path / f"{method}.wav"
    arguments = [
        "sonify",
        "--input",
        str(source),
        "--output",
        str(output),
        "--duration",
        "0.01",
        "--method",
        method,
    ]
    for setting in method_settings:
        arguments.extend(("--method-param", setting))

    result = runner.invoke(main, arguments)

    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert output.is_file()


def test_expected_library_error_is_reported_without_traceback(runner, spectrum_file, tmp_path):
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


@pytest.mark.parametrize(
    ("option", "setting", "message"),
    [
        ("--preprocess", "unknown=8", "unknown preprocessing"),
        ("--method-param", "unknown=8", "unknown parameter"),
        ("--preprocess", "scale_statistic", "key=value"),
    ],
)
def test_generic_setting_errors_are_actionable(
    runner, spectrum_file, tmp_path, option, setting, message
):
    result = runner.invoke(
        main,
        [
            "sonify",
            "--input",
            str(spectrum_file),
            "--output",
            str(tmp_path / "out.wav"),
            "--duration",
            "0.01",
            "--method",
            "erb",
            option,
            setting,
        ],
    )

    assert result.exit_code != 0
    assert message in result.output


def test_duplicate_generic_setting_is_rejected(runner, profile_file, tmp_path):
    result = runner.invoke(
        main,
        [
            "sonify",
            "--input",
            str(profile_file),
            "--output",
            str(tmp_path / "out.wav"),
            "--duration",
            "0.01",
            "--method-param",
            "sr=8000",
            "--method-param",
            "sr=16000",
        ],
    )

    assert result.exit_code != 0
    assert "repeats setting 'sr'" in result.output


@pytest.mark.parametrize(
    ("fixture_name", "extra"),
    [
        ("profile_file", ["--repeat", "1", "--method-param", "sr=8000"]),
        (
            "spectrum_file",
            [
                "--method",
                "erb",
                "--method-param",
                "sr=8000",
                "--method-param",
                "max_freq=3000",
                "--method-param",
                "n_bands=4",
            ],
        ),
        (
            "layered_file",
            [
                "--method-param",
                "sr=8000",
                "--method-param",
                "max_freq=3000",
                "--method-param",
                "n_bands=4",
            ],
        ),
    ],
)
def test_sonify_command_covers_every_dimensionality(runner, request, tmp_path, fixture_name, extra):
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
            "0.04",
            "--speed",
            "2",
            "--preprocess",
            "scale_statistic='mad'",
            "--preprocess",
            "clip_percentiles=(5, 95)",
            *extra,
        ],
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert output.is_file()


def test_sonify_command_accepts_axes_and_layer_rebin(runner, tmp_path):
    source = tmp_path / "channels_last.npy"
    np.save(source, np.random.default_rng(1).random((8, 6, 4)))
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
            "0.01",
            "--layer-axis",
            "2",
            "--preprocess",
            "layer_rebin=2",
            "--method-param",
            "sr=8000",
            "--method-param",
            "max_freq=3000",
            "--method-param",
            "n_bands=4",
        ],
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert output.is_file()


def test_missing_input_and_invalid_postprocessor_setting_fail_at_the_boundary(
    runner, profile_file, tmp_path
):
    missing = runner.invoke(
        main,
        [
            "sonify",
            "--input",
            str(tmp_path / "missing.npy"),
            "--output",
            str(tmp_path / "out.wav"),
            "--duration",
            "1",
        ],
    )
    invalid_decoder = runner.invoke(
        main,
        [
            "sonify",
            "--input",
            str(profile_file),
            "--output",
            str(tmp_path / "musicnet.wav"),
            "--duration",
            "1",
            "--postprocess",
            "musicnet",
            "--postprocess-param",
            "decoder_id=6",
        ],
    )

    assert missing.exit_code != 0
    assert "does not exist" in missing.output
    assert invalid_decoder.exit_code != 0
    assert "decoder_id" in invalid_decoder.output


def test_list_methods_names_optional_extras(runner):
    result = runner.invoke(main, ["list-methods"])

    assert result.exit_code == 0
    for extra in ("hifigan", "musicnet", "rave"):
        assert f"pip install radiosonify[{extra}]" in result.output
    erb_line = next(line for line in result.output.splitlines() if line.strip().startswith("erb "))
    assert "pip install" not in erb_line


def test_list_settings_prints_the_complete_registered_surface(runner):
    from radiosonify._perceptual_config import EVENT_DEFAULTS, VOICE_DEFAULTS
    from radiosonify.preprocessing import preprocessing_defaults
    from radiosonify.registry import available_methods, available_postprocessors

    result = runner.invoke(main, ["list-settings"])

    assert result.exit_code == 0
    for name, value in preprocessing_defaults().items():
        assert name in result.output
        assert f"default: {value!r}" in result.output
    for spec in (*available_methods(), *available_postprocessors()):
        for name, value in spec.defaults.items():
            assert name in result.output
            assert f"default: {value!r}" in result.output
    for name, value in (*VOICE_DEFAULTS.items(), *EVENT_DEFAULTS.items()):
        assert name in result.output
        assert f"default: {value!r}" in result.output
    assert "voice_params accepts:" in result.output
    assert "event_params accepts:" in result.output
    assert "(none)" in result.output
