import numpy as np
import pytest

import astrosonify.musicnet as musicnet_module


class _FakeCuda:
    @staticmethod
    def is_available():
        return False


class _FakeTorch:
    cuda = _FakeCuda()


def test_musicnet_requires_cuda(monkeypatch):
    monkeypatch.setattr(musicnet_module, "_require_torch", lambda: _FakeTorch)
    monkeypatch.setattr(musicnet_module, "_require_tqdm", lambda: object())

    data = np.random.default_rng(42).random(100).astype(np.float32)
    with pytest.raises(RuntimeError, match="CUDA"):
        musicnet_module.musicnet(data)


def test_musicnet_decoder_id_validation(monkeypatch):
    monkeypatch.setattr(musicnet_module, "_require_torch", lambda: _FakeTorch)
    monkeypatch.setattr(musicnet_module, "_require_tqdm", lambda: object())

    with pytest.raises(ValueError, match="decoder_id"):
        musicnet_module.musicnet(np.zeros(10, dtype=np.float32), decoder_id=6)
