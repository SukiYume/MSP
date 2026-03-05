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
        assert "astronify" in result.output

    def test_astronify_command(self, runner, profile_file, tmp_path, monkeypatch):
        out = str(tmp_path / "out.wav")

        def _fake_astronify(data, note_spacing=0.01, time_downsample=10, output=None):
            import soundfile as sf
            sf.write(output, np.zeros(480, dtype=np.float32), 48000)
            return np.zeros(480, dtype=np.float32), 48000

        from astrosonify import astronify_method
        monkeypatch.setattr(astronify_method, "astronify_sonify", _fake_astronify)

        result = runner.invoke(main, [
            "astronify", "--input", profile_file, "--output", out,
            "--note-spacing", "0.02", "--downsample", "5"
        ])
        assert result.exit_code == 0, f"Failed: {result.output}\n{result.exception}"

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

    def test_musicnet_decoder_id_range_validation(self, runner, tmp_path):
        in_wav = str(tmp_path / "in.wav")
        out = str(tmp_path / "out.wav")
        np.save(str(tmp_path / "dummy.npy"), np.zeros(8, dtype=np.float32))
        with open(in_wav, "wb") as f:
            f.write(b"RIFF")

        result = runner.invoke(
            main,
            ["musicnet", "--input", in_wav, "--output", out, "--decoder-id", "6"],
        )
        assert result.exit_code != 0
        assert "not in the range" in result.output

    def test_profile_duration_range_validation(self, runner, profile_file, tmp_path):
        out = str(tmp_path / "out.wav")
        result = runner.invoke(
            main,
            ["profile", "--input", profile_file, "--output", out, "--duration", "-1"],
        )
        assert result.exit_code != 0
        assert "not in the range" in result.output
