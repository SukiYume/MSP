import builtins

import pytest

from radiosonify.runtime import _temporary_torch_seed, require


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _Generator:
    def __init__(self):
        self.seeds = []

    def manual_seed(self, seed):
        self.seeds.append(seed)


class _Random:
    def __init__(self):
        self.default_generator = _Generator()
        self.forks = []

    def fork_rng(self, **kwargs):
        self.forks.append(kwargs)
        return _Context()


class _DeviceModule:
    def __init__(self, current=3):
        self.current = current
        self.seeds = []
        self.contexts = []

    def current_device(self):
        return self.current

    def device(self, index):
        self.contexts.append(index)
        return _Context()

    def manual_seed(self, seed):
        self.seeds.append(seed)


class _Torch:
    def __init__(self):
        self.random = _Random()
        self.cuda = _DeviceModule()
        self.mps = _DeviceModule()


def test_require_reports_broken_optional_binary(monkeypatch):
    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "torch":
            raise OSError("shm.dll could not be loaded")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(ImportError, match="installed but failed to load.*shm.dll"):
        require("torch", "hifigan")


def test_require_reports_missing_optional_dependency(monkeypatch):
    original_import = builtins.__import__

    def missing_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("missing package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_import)
    with pytest.raises(ImportError, match=r"radiosonify\[rave\].*missing package"):
        require("torch", "rave")


def test_temporary_torch_seed_supports_none_and_cpu():
    torch = _Torch()

    with _temporary_torch_seed(torch, None):
        pass
    with _temporary_torch_seed(torch, 17):
        pass

    assert torch.random.forks == [{"devices": []}]
    assert torch.random.default_generator.seeds == [17]


def test_temporary_torch_seed_resolves_cuda_index_and_mps_default():
    torch = _Torch()

    with _temporary_torch_seed(torch, 4, device="cuda:2"):
        pass
    with _temporary_torch_seed(torch, 5, device="mps"):
        pass

    assert torch.random.forks == [
        {"devices": [2], "device_type": "cuda"},
        {"devices": [0], "device_type": "mps"},
    ]
    assert torch.cuda.contexts == [2]
    assert torch.cuda.seeds == [4]
    assert torch.mps.seeds == [5]


def test_temporary_torch_seed_rejects_unknown_device():
    with pytest.raises(ValueError, match="unsupported Torch RNG device"):
        with _temporary_torch_seed(_Torch(), 3, device="tpu"):
            pass
