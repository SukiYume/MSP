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


def test_griffinlim_legacy_n_mels_alias(runner, spectrum_file, tmp_path):
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
            "--n-mels",
            "16",
            "--time-rebin",
            "8",
        ],
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert output.is_file()


def test_griffinlim_rejects_both_frequency_aliases(runner, spectrum_file, tmp_path):
    result = runner.invoke(
        main,
        [
            "griffinlim",
            "--input",
            str(spectrum_file),
            "--output",
            str(tmp_path / "out.wav"),
            "--n-mels",
            "8",
            "--freq-rebin",
            "8",
        ],
    )

    assert result.exit_code != 0
    assert "cannot be supplied together" in result.output


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
