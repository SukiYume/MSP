"""Optional-runtime loading and temporary runtime state."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


def require(module: str, extra: str) -> Any:
    """Import an optional dependency with an actionable installation error."""
    try:
        return __import__(module, fromlist=["*"])
    except ImportError as err:
        raise ImportError(
            f"This feature requires '{module}'. Install with: "
            f"pip install radiosonify[{extra}]. Original error: {err}"
        ) from err
    except OSError as err:
        raise ImportError(
            f"Optional dependency '{module}' is installed but failed to load its binary "
            "libraries. Repair or reinstall that package in the active environment. "
            f"Original error: {err}"
        ) from err


@contextmanager
def _temporary_torch_seed(
    torch: Any,
    seed: int | None,
    *,
    device: Any | None = None,
):
    """Apply a Torch seed while preserving the caller's RNG state."""
    if seed is None:
        yield
        return

    if device is None:
        device_type = "cpu"
        device_index = None
    else:
        device_text = str(device)
        device_type = getattr(device, "type", device_text.split(":", 1)[0])
        device_index = None if isinstance(device, str) else getattr(device, "index", None)
        if device_index is None and ":" in device_text:
            device_index = int(device_text.split(":", 1)[1])

    if device_type == "cpu":
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            yield
        return
    if device_type not in {"cuda", "mps"}:
        raise ValueError(f"unsupported Torch RNG device: {device_type}")

    device_module = getattr(torch, device_type)
    if device_index is None:
        device_index = int(device_module.current_device()) if device_type == "cuda" else 0
    with torch.random.fork_rng(devices=[device_index], device_type=device_type):
        if device_type == "cuda":
            with device_module.device(device_index):
                device_module.manual_seed(seed)
                yield
        else:
            device_module.manual_seed(seed)
            yield
