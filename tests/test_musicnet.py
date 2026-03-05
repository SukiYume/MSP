import numpy as np
import pathlib
import pytest

import radiosonify.musicnet as musicnet_module


class _FakeCuda:
    @staticmethod
    def is_available():
        return False


class _FakeTorch:
    cuda = _FakeCuda()

    @staticmethod
    def device(name):
        return name


def test_musicnet_does_not_require_cuda_and_keeps_pathlib_intact(monkeypatch):
    original_posix = pathlib.PosixPath

    monkeypatch.setattr(musicnet_module, "_require_torch", lambda: _FakeTorch)
    monkeypatch.setattr(musicnet_module, "_require_tqdm", lambda: object())
    monkeypatch.setattr(
        musicnet_module,
        "get_model_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop-here")),
    )

    data = np.random.default_rng(42).random(100).astype(np.float32)
    with pytest.raises(RuntimeError, match="stop-here"):
        musicnet_module.musicnet(data)

    assert pathlib.PosixPath is original_posix


def test_musicnet_decoder_id_validation(monkeypatch):
    monkeypatch.setattr(musicnet_module, "_require_torch", lambda: _FakeTorch)
    monkeypatch.setattr(musicnet_module, "_require_tqdm", lambda: object())

    with pytest.raises(ValueError, match="decoder_id"):
        musicnet_module.musicnet(np.zeros(10, dtype=np.float32), decoder_id=6)
