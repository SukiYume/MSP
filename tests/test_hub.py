import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import numpy as np
import pytest
from huggingface_hub.utils import LocalEntryNotFoundError

from radiosonify.hub import (
    REVISION,
    _write_pcm16_atomic,
    get_data_path,
    get_instrument_path,
    get_model_path,
    load_example,
)

REPO_ID = "TorchLight/radiosonify"


class TestGetDataPath:
    @patch("radiosonify.hub.hf_hub_download")
    def test_calls_hf_download(self, mock_download):
        mock_download.return_value = "/fake/path/Burst.npy"
        result = get_data_path("Burst.npy")
        mock_download.assert_called_once()
        assert mock_download.call_args.kwargs["revision"] == REVISION
        assert result == "/fake/path/Burst.npy"

    @patch("radiosonify.hub.hf_hub_download")
    def test_returns_path_string(self, mock_download):
        mock_download.return_value = "/fake/path/Burst.npy"
        result = get_data_path("Burst.npy")
        assert isinstance(result, str)

    @patch("radiosonify.hub.hf_hub_download")
    @patch("radiosonify.hub.time.sleep")
    def test_retries_online_local_entry_error(self, mock_sleep, mock_download):
        mock_download.side_effect = [
            LocalEntryNotFoundError("not in cache"),
            LocalEntryNotFoundError("network unavailable"),
            "/fake/path/Burst.npy",
        ]

        assert get_data_path("Burst.npy") == "/fake/path/Burst.npy"
        assert mock_download.call_count == 3
        mock_sleep.assert_called_once_with(0.3)

    @patch("radiosonify.hub.hf_hub_download")
    @patch("radiosonify.hub.time.sleep")
    def test_wraps_download_error_with_actionable_message(self, mock_sleep, mock_download):
        mock_download.side_effect = [
            LocalEntryNotFoundError("not in cache"),
            LocalEntryNotFoundError("network down"),
            LocalEntryNotFoundError("still offline"),
        ]

        with pytest.raises(RuntimeError, match="Failed to download") as error:
            get_data_path("Burst.npy")

        assert "not found in Hugging Face repo" not in str(error.value)
        assert mock_download.call_count == 3
        mock_sleep.assert_called_once_with(0.3)


class TestGetModelPath:
    @patch("radiosonify.hub.hf_hub_download")
    def test_hifigan_model(self, mock_download):
        mock_download.return_value = "/fake/path/generator.pth"
        result = get_model_path("hifigan", "generator.pth")
        mock_download.assert_called_once()
        assert result == "/fake/path/generator.pth"


class TestGetInstrumentPath:
    def test_atomic_writer_uses_unique_temporary_files_across_threads(self, tmp_path):
        destination = tmp_path / "violin.wav"
        audio = np.sin(np.linspace(0, 4 * np.pi, 4_800)).astype(np.float32)
        barrier = Barrier(8)

        def write_once(_index):
            barrier.wait()
            _write_pcm16_atomic(destination, audio)

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(write_once, range(8)))

        assert destination.is_file()
        assert not list(tmp_path.glob(".violin.wav.*.tmp"))

    @patch("radiosonify.hub.hf_hub_download")
    def test_violin_is_generated_locally_without_a_download(
        self, mock_download, monkeypatch, tmp_path
    ):
        monkeypatch.setattr("radiosonify.hub.CACHE_DIR", str(tmp_path))
        result = get_instrument_path("violin")
        generated = Path(result)

        mock_download.assert_not_called()
        assert generated.is_file()
        assert generated.name == "violin.wav"
        with wave.open(str(generated), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == 48_000
            assert wav.getnframes() > 0

    def test_generated_instrument_is_deterministic_and_reuses_the_cache(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr("radiosonify.hub.CACHE_DIR", str(tmp_path))
        first = Path(get_instrument_path("piano"))
        first_bytes = first.read_bytes()
        first_mtime = first.stat().st_mtime_ns

        second = Path(get_instrument_path("piano"))

        assert second == first
        assert second.read_bytes() == first_bytes
        assert second.stat().st_mtime_ns == first_mtime

    def test_unknown_instrument_raises(self):
        with pytest.raises(ValueError, match="Unknown instrument"):
            get_instrument_path("drums")


class TestLoadExample:
    @patch("radiosonify.hub.np.load")
    @patch("radiosonify.hub.get_data_path")
    def test_load_burst(self, mock_get_path, mock_np_load):
        mock_get_path.return_value = "/fake/Burst.npy"
        mock_np_load.return_value = "fake_array"
        result = load_example("burst")
        mock_get_path.assert_called_once_with("Burst.npy")
        assert result == "fake_array"

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            load_example("nonexistent")


def test_cache_directory_follows_a_later_environment_change(monkeypatch, tmp_path):
    from radiosonify import hub

    monkeypatch.delenv("RADIOSONIFY_CACHE_DIR", raising=False)
    assert hub._cache_dir() == hub.CACHE_DIR

    # The variable is read on every call, so a process that configures it after
    # importing the package still redirects downloads and generated instruments.
    monkeypatch.setenv("RADIOSONIFY_CACHE_DIR", str(tmp_path))
    assert hub._cache_dir() == str(tmp_path)
    assert Path(hub.get_instrument_path("violin")).is_relative_to(tmp_path)
