import pytest
from unittest.mock import patch, MagicMock
from astrosonify.hub import get_data_path, get_model_path, get_instrument_path, load_example

REPO_ID = "TorchLight/astrosonify"


class TestGetDataPath:
    @patch("astrosonify.hub.hf_hub_download")
    def test_calls_hf_download(self, mock_download):
        mock_download.return_value = "/fake/path/Burst.npy"
        result = get_data_path("Burst.npy")
        mock_download.assert_called_once()
        assert result == "/fake/path/Burst.npy"

    @patch("astrosonify.hub.hf_hub_download")
    def test_returns_path_string(self, mock_download):
        mock_download.return_value = "/fake/path/Burst.npy"
        result = get_data_path("Burst.npy")
        assert isinstance(result, str)

    @patch("astrosonify.hub.hf_hub_download")
    def test_wraps_download_error_with_actionable_message(self, mock_download):
        mock_download.side_effect = RuntimeError("network down")
        with pytest.raises(RuntimeError, match="Failed to download"):
            get_data_path("Burst.npy")


class TestGetModelPath:
    @patch("astrosonify.hub.hf_hub_download")
    def test_hifigan_model(self, mock_download):
        mock_download.return_value = "/fake/path/generator.pth"
        result = get_model_path("hifigan", "generator.pth")
        mock_download.assert_called_once()
        assert result == "/fake/path/generator.pth"


class TestGetInstrumentPath:
    @patch("astrosonify.hub.hf_hub_download")
    def test_violin(self, mock_download):
        mock_download.return_value = "/fake/path/vio.wav"
        result = get_instrument_path("violin")
        mock_download.assert_called_once()
        assert result == "/fake/path/vio.wav"

    def test_unknown_instrument_raises(self):
        with pytest.raises(ValueError, match="Unknown instrument"):
            get_instrument_path("drums")


class TestLoadExample:
    @patch("astrosonify.hub.np.load")
    @patch("astrosonify.hub.get_data_path")
    def test_load_burst(self, mock_get_path, mock_np_load):
        mock_get_path.return_value = "/fake/Burst.npy"
        mock_np_load.return_value = "fake_array"
        result = load_example("burst")
        mock_get_path.assert_called_once_with("Burst.npy")
        assert result == "fake_array"

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            load_example("nonexistent")
