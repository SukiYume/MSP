import numpy as np
import soundfile as sf

from astrosonify.astronify_method import astronify_sonify


class _FakeSoniSeries:
    def __init__(self, data_table):
        self.data_table = data_table
        self.note_spacing = 0.01

    def sonify(self):
        return None

    def write(self, path):
        audio = np.sin(np.linspace(0, 2 * np.pi * 2, 800)).astype(np.float32)
        sf.write(path, audio, 8000)


def _fake_require_astronify():
    return dict, _FakeSoniSeries


def test_astronify_sonify_returns_audio(monkeypatch):
    monkeypatch.setattr("astrosonify.astronify_method._require_astronify", _fake_require_astronify)

    data = np.random.default_rng(42).random(128)
    audio, sr = astronify_sonify(data, note_spacing=0.02, time_downsample=4)

    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert sr == 8000


def test_astronify_sonify_saves_output(monkeypatch, tmp_path):
    monkeypatch.setattr("astrosonify.astronify_method._require_astronify", _fake_require_astronify)

    out = tmp_path / "astronify.wav"
    data = np.random.default_rng(42).random((64, 8))
    astronify_sonify(data, output=str(out))

    assert out.exists()
    assert out.stat().st_size > 0
