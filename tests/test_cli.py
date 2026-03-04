import numpy as np
import pytest
from click.testing import CliRunner
from astrosonify.cli import main


LEGACY_GRIFFIN_TIME_BINS = 128
LEGACY_GRIFFIN_FREQ_BINS = 512


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def npy_file(tmp_path):
    data = np.random.default_rng(42).random((256, 1024))
    path = tmp_path / "test.npy"
    np.save(str(path), data)
    return str(path)


@pytest.fixture
def profile_file(tmp_path):
    data = np.random.default_rng(42).random(200)
    path = tmp_path / "profile.npy"
    np.save(str(path), data)
    return str(path)


class TestCLI:
    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "AstroSonify" in result.output

    def test_list_methods(self, runner):
        result = runner.invoke(main, ["list-methods"])
        assert result.exit_code == 0
        assert "griffinlim" in result.output
        assert "profile" in result.output

    def test_profile_command(self, runner, profile_file, tmp_path):
        out = str(tmp_path / "out.wav")
        result = runner.invoke(main, [
            "profile", "--input", profile_file, "--output", out,
            "--duration", "0.5", "--no-instrument"
        ])
        assert result.exit_code == 0, f"Failed: {result.output}\n{result.exception}"

    def test_amplitude_command(self, runner, profile_file, tmp_path):
        out = str(tmp_path / "out.wav")
        result = runner.invoke(main, [
            "amplitude", "--input", profile_file, "--output", out,
            "--duration", "0.5"
        ])
        assert result.exit_code == 0, f"Failed: {result.output}\n{result.exception}"

    def test_griffinlim_command(self, runner, npy_file, tmp_path):
        out = str(tmp_path / "out.wav")
        result = runner.invoke(main, [
            "griffinlim", "--input", npy_file, "--output", out,
            "--n-iter", "5", "--n-mels", str(LEGACY_GRIFFIN_FREQ_BINS),
            "--time-rebin", str(LEGACY_GRIFFIN_TIME_BINS),
            "--freq-rebin", str(LEGACY_GRIFFIN_FREQ_BINS),
        ])
        assert result.exit_code == 0, f"Failed: {result.output}\n{result.exception}"

    def test_missing_input(self, runner, tmp_path):
        out = str(tmp_path / "out.wav")
        result = runner.invoke(main, ["griffinlim", "--output", out])
        assert result.exit_code != 0
