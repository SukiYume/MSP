import numpy as np
import pytest
import soundfile as sf

from radiosonify.audio_io import save_audio


def test_writes_pcm16_wav(tmp_path):
    path = tmp_path / "test.wav"
    save_audio(np.sin(np.linspace(0, 2 * np.pi, 48000)).astype(np.float32), 48000, path)
    assert path.exists()
    assert sf.info(path).subtype == "PCM_16"


def test_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "test.wav"
    save_audio(np.zeros(16, dtype=np.float32), 48000, path)
    assert path.exists()


def test_writes_samples_by_channels_stereo(tmp_path):
    path = tmp_path / "stereo.wav"
    stereo = np.column_stack((np.linspace(-0.5, 0.5, 32), np.linspace(0.5, -0.5, 32)))
    save_audio(stereo, 8_000, path)
    data, sr = sf.read(path, always_2d=True)
    assert data.shape == (32, 2)
    assert sr == 8_000


def test_rejects_clipping(tmp_path):
    with pytest.raises(ValueError, match="clipping"):
        save_audio(np.array([0.0, 1.1]), 48000, tmp_path / "bad.wav")


def test_rejects_non_wav_path_before_writing(tmp_path):
    with pytest.raises(ValueError, match=r"\.wav"):
        save_audio(np.zeros(16), 48000, tmp_path / "bad.flac")
    assert not (tmp_path / "bad.flac").exists()
