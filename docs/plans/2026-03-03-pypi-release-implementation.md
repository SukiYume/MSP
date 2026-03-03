# AstroSonify PyPI Release Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure MSP from standalone research scripts into an installable Python package `astrosonify` with API + CLI, published to PyPI.

**Architecture:** Flat module structure under `src/astrosonify/`. Each sonification method is one module exposing a single public function. Shared utilities in `core.py`. Models/data hosted on Hugging Face Hub, downloaded lazily. CLI built with click.

**Tech Stack:** Python 3.9+, numpy, scipy, librosa, soundfile, click, huggingface_hub, pytest. Optional: torch, scikit-image, astropy, astronify, tqdm.

---

### Task 1: Create package scaffolding and pyproject.toml

**Files:**
- Create: `src/astrosonify/__init__.py`
- Create: `src/astrosonify/models/__init__.py`
- Create: `src/astrosonify/models/hifigan/__init__.py`
- Create: `src/astrosonify/models/musicnet/__init__.py`
- Create: `pyproject.toml`
- Create: `tests/__init__.py`

**Step 1: Create directory structure**

```bash
mkdir -p src/astrosonify/models/hifigan
mkdir -p src/astrosonify/models/musicnet
mkdir -p tests
```

**Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "astrosonify"
version = "0.1.0"
description = "Methods for Sonifying Pulse - Convert radio telescope data into audible sound"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.9"
authors = [
    {name = "XiaoQing", email = ""},
]
keywords = ["astronomy", "sonification", "radio-pulse", "audio", "FRB"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Astronomy",
    "Topic :: Multimedia :: Sound/Audio",
]
dependencies = [
    "numpy",
    "scipy",
    "librosa",
    "soundfile",
    "huggingface_hub",
    "click",
]

[project.optional-dependencies]
astronify = ["astropy", "astronify"]
hifigan = ["torch", "scikit-image"]
musicnet = ["torch", "tqdm"]
all = ["astropy", "astronify", "torch", "scikit-image", "tqdm"]
dev = ["pytest", "pytest-cov"]

[project.scripts]
astrosonify = "astrosonify.cli:main"

[project.urls]
Homepage = "https://github.com/SukiYume/MSP"
Repository = "https://github.com/SukiYume/MSP"

[tool.hatch.build.targets.sdist]
include = ["src/astrosonify"]

[tool.hatch.build.targets.wheel]
packages = ["src/astrosonify"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 3: Create `src/astrosonify/__init__.py`**

```python
"""AstroSonify - Methods for Sonifying Pulse."""

__version__ = "0.1.0"
```

Note: Public API imports will be added in later tasks as each method is implemented.

**Step 4: Create empty `__init__.py` files**

Create empty `__init__.py` in:
- `src/astrosonify/models/__init__.py`
- `src/astrosonify/models/hifigan/__init__.py`
- `src/astrosonify/models/musicnet/__init__.py`
- `tests/__init__.py`

**Step 5: Verify the package can be installed in dev mode**

Run: `pip install -e ".[dev]"`
Expected: installs successfully, `python -c "import astrosonify; print(astrosonify.__version__)"` prints `0.1.0`

**Step 6: Commit**

```bash
git add pyproject.toml src/ tests/__init__.py
git commit -m "feat: create package scaffolding with pyproject.toml"
```

---

### Task 2: Implement core.py - shared utilities

**Files:**
- Create: `src/astrosonify/core.py`
- Create: `tests/test_core.py`

**Step 1: Write failing tests for core utilities**

```python
# tests/test_core.py
import numpy as np
import pytest
from astrosonify.core import (
    normalize,
    del_burst,
    rebin_spectrogram,
    to_profile,
    save_audio,
)


class TestNormalize:
    def test_output_range_0_to_1(self):
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = normalize(data)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_constant_array(self):
        data = np.ones(10) * 5.0
        result = normalize(data)
        assert np.all(result == 0.0)

    def test_2d_array(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = normalize(data)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)


class TestDelBurst:
    def test_output_range(self):
        rng = np.random.default_rng(42)
        data = rng.random((100, 50)) * 100 + 1
        result = del_burst(data, exposure_cut=25)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_shape_preserved(self):
        rng = np.random.default_rng(42)
        data = rng.random((100, 50)) * 100 + 1
        result = del_burst(data)
        assert result.shape == (100, 50)


class TestRebinSpectrogram:
    def test_downsample_both_axes(self):
        data = np.ones((100, 200))
        result = rebin_spectrogram(data, time_bins=50, freq_bins=100)
        assert result.shape == (50, 100)

    def test_none_keeps_original(self):
        data = np.ones((100, 200))
        result = rebin_spectrogram(data, time_bins=None, freq_bins=None)
        assert result.shape == (100, 200)

    def test_values_averaged(self):
        data = np.arange(12).reshape(4, 3).astype(float)
        result = rebin_spectrogram(data, time_bins=2, freq_bins=None)
        assert result.shape == (2, 3)
        # First bin: mean of rows 0,1; second bin: mean of rows 2,3
        np.testing.assert_array_almost_equal(result[0], [1.5, 2.5, 3.5])
        np.testing.assert_array_almost_equal(result[1], [7.5, 8.5, 9.5])

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2D"):
            rebin_spectrogram(np.ones(10), time_bins=5)


class TestToProfile:
    def test_2d_to_1d(self):
        data = np.ones((100, 50))
        result = to_profile(data)
        assert result.ndim == 1
        assert len(result) == 100

    def test_1d_passthrough(self):
        data = np.ones(100)
        result = to_profile(data)
        assert result.ndim == 1
        assert len(result) == 100

    def test_downsample(self):
        data = np.ones(100)
        result = to_profile(data, downsample=10)
        assert len(result) == 10


class TestSaveAudio:
    def test_writes_wav(self, tmp_path):
        audio = np.sin(np.linspace(0, 2 * np.pi, 48000)).astype(np.float32)
        path = tmp_path / "test.wav"
        save_audio(audio, 48000, str(path))
        assert path.exists()
        assert path.stat().st_size > 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_core.py -v`
Expected: FAIL (ImportError - module not found)

**Step 3: Implement core.py**

```python
# src/astrosonify/core.py
"""Shared utilities for AstroSonify."""

import numpy as np
import soundfile as sf


def normalize(data: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    data = data.astype(np.float64)
    dmin, dmax = data.min(), data.max()
    if dmax == dmin:
        return np.zeros_like(data)
    return (data - dmin) / (dmax - dmin)


def del_burst(data: np.ndarray, exposure_cut: int = 25) -> np.ndarray:
    """Clean burst data by clipping outliers and normalizing.

    Divides by column mean, clips to percentile range, normalizes to [0, 1].

    Args:
        data: 2D array (time x freq).
        exposure_cut: Percentile cut parameter. Values outside
            [1/exposure_cut, (exposure_cut-1)/exposure_cut] are clipped.

    Returns:
        Cleaned and normalized 2D array.
    """
    data = data.astype(np.float64)
    h, w = data.shape
    col_mean = np.mean(data, axis=0)
    col_mean[col_mean == 0] = 1.0
    data = data / col_mean
    flat = np.sort(data.flatten())
    vmin = flat[int(h * w / exposure_cut)]
    vmax = flat[int(h * w / exposure_cut * (exposure_cut - 1))]
    data = np.clip(data, vmin, vmax)
    return normalize(data)


def rebin_spectrogram(
    data: np.ndarray,
    time_bins: int | None = None,
    freq_bins: int | None = None,
) -> np.ndarray:
    """Rebin a 2D spectrogram by averaging adjacent bins.

    Args:
        data: 2D array (time x freq).
        time_bins: Target number of time bins. None keeps original.
        freq_bins: Target number of freq bins. None keeps original.

    Returns:
        Rebinned 2D array.
    """
    if data.ndim != 2:
        raise ValueError("rebin_spectrogram requires 2D input array")

    result = data.astype(np.float64)

    if time_bins is not None and time_bins != result.shape[0]:
        t = result.shape[0]
        usable = (t // time_bins) * time_bins
        result = result[:usable].reshape(time_bins, -1, result.shape[1]).mean(axis=1)

    if freq_bins is not None and freq_bins != result.shape[1]:
        f = result.shape[1]
        usable = (f // freq_bins) * freq_bins
        result = result[:, :usable].reshape(result.shape[0], freq_bins, -1).mean(axis=2)

    return result


def to_profile(
    data: np.ndarray,
    downsample: int | None = None,
) -> np.ndarray:
    """Convert data to 1D pulse profile.

    If 2D, averages along frequency axis (axis=1).
    Optionally downsamples by averaging adjacent bins.

    Args:
        data: 1D profile or 2D spectrogram (time x freq).
        downsample: Factor to downsample by. None keeps original.

    Returns:
        1D profile array.
    """
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    elif data.ndim != 1:
        raise ValueError("to_profile expects 1D or 2D input")

    data = data.astype(np.float64)

    if downsample is not None and downsample > 1:
        usable = (len(data) // downsample) * downsample
        data = data[:usable].reshape(-1, downsample).mean(axis=1)

    return data


def save_audio(audio: np.ndarray, sr: int, path: str) -> None:
    """Save audio array to WAV file.

    Args:
        audio: 1D audio array.
        sr: Sample rate.
        path: Output file path.
    """
    sf.write(path, audio, sr)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/astrosonify/core.py tests/test_core.py
git commit -m "feat: implement core utilities (normalize, del_burst, rebin, to_profile, save_audio)"
```

---

### Task 3: Implement hub.py - Hugging Face download manager

**Files:**
- Create: `src/astrosonify/hub.py`
- Create: `tests/test_hub.py`

**Step 1: Write failing tests**

```python
# tests/test_hub.py
import pytest
from unittest.mock import patch, MagicMock
from astrosonify.hub import get_data_path, get_model_path, load_example

REPO_ID = "SukiYume/astrosonify"


class TestGetDataPath:
    @patch("astrosonify.hub.hf_hub_download")
    def test_calls_hf_download(self, mock_download):
        mock_download.return_value = "/fake/path/Burst.npy"
        result = get_data_path("Burst.npy")
        mock_download.assert_called_once_with(
            repo_id=REPO_ID,
            filename="data/Burst.npy",
            cache_dir=pytest.approx(result, abs=100),  # just check it returns
        )

    @patch("astrosonify.hub.hf_hub_download")
    def test_returns_path_string(self, mock_download):
        mock_download.return_value = "/fake/path/Burst.npy"
        result = get_data_path("Burst.npy")
        assert result == "/fake/path/Burst.npy"


class TestGetModelPath:
    @patch("astrosonify.hub.hf_hub_download")
    def test_hifigan_model(self, mock_download):
        mock_download.return_value = "/fake/path/generator.pth"
        result = get_model_path("hifigan", "generator.pth")
        mock_download.assert_called_once()
        assert result == "/fake/path/generator.pth"


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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hub.py -v`
Expected: FAIL

**Step 3: Implement hub.py**

```python
# src/astrosonify/hub.py
"""Hugging Face Hub download manager for models and example data."""

import os
import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = "SukiYume/astrosonify"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "astrosonify")

EXAMPLE_MAP = {
    "burst": "Burst.npy",
    "raw_burst": "RawBurst.npy",
    "parkes_burst": "ParkesBurst.npy",
    "profile": "Profile.npy",
}

INSTRUMENT_MAP = {
    "violin": "vio.wav",
    "piano": "piano.wav",
}


def get_data_path(filename: str) -> str:
    """Download example data file from HF Hub and return local path."""
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=f"data/{filename}",
        cache_dir=CACHE_DIR,
    )


def get_model_path(model_name: str, filename: str) -> str:
    """Download model file from HF Hub and return local path."""
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=f"models/{model_name}/{filename}",
        cache_dir=CACHE_DIR,
    )


def get_instrument_path(name: str) -> str:
    """Download instrument sample from HF Hub and return local path."""
    if name not in INSTRUMENT_MAP:
        raise ValueError(f"Unknown instrument: {name}. Available: {list(INSTRUMENT_MAP.keys())}")
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=f"instruments/{INSTRUMENT_MAP[name]}",
        cache_dir=CACHE_DIR,
    )


def load_example(name: str) -> np.ndarray:
    """Load an example dataset by name.

    Args:
        name: One of 'burst', 'raw_burst', 'parkes_burst', 'profile'.

    Returns:
        numpy array with the example data.
    """
    if name not in EXAMPLE_MAP:
        raise ValueError(f"Unknown example: {name}. Available: {list(EXAMPLE_MAP.keys())}")
    path = get_data_path(EXAMPLE_MAP[name])
    return np.load(path)
```

**Step 4: Run tests**

Run: `pytest tests/test_hub.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/astrosonify/hub.py tests/test_hub.py
git commit -m "feat: implement HF Hub download manager for models and data"
```

---

### Task 4: Implement profile.py - Method 1 (pulse profile to waveform)

**Files:**
- Create: `src/astrosonify/profile.py`
- Create: `tests/test_profile.py`

**Step 1: Write failing tests**

```python
# tests/test_profile.py
import numpy as np
import pytest
from unittest.mock import patch
from astrosonify.profile import profile_to_wave


class TestProfileToWave:
    def test_returns_tuple(self):
        data = np.random.default_rng(42).random(100)
        audio, sr = profile_to_wave(data, sr=48000, duration=1)
        assert isinstance(audio, np.ndarray)
        assert sr == 48000

    def test_output_length_matches_duration(self):
        data = np.random.default_rng(42).random(100)
        audio, sr = profile_to_wave(data, sr=48000, duration=2)
        assert len(audio) == 48000 * 2

    def test_2d_input_auto_averages(self):
        data = np.random.default_rng(42).random((100, 50))
        audio, sr = profile_to_wave(data, sr=48000, duration=1)
        assert audio.ndim == 1

    def test_no_instrument(self):
        data = np.random.default_rng(42).random(100)
        audio, sr = profile_to_wave(data, sr=48000, duration=1, instrument=None)
        assert len(audio) == 48000

    @patch("astrosonify.profile.get_instrument_path")
    def test_with_instrument(self, mock_get_path, tmp_path):
        # Create a fake instrument WAV
        import soundfile as sf
        fake_wav = np.sin(np.linspace(0, 2 * np.pi * 440, 4800)).astype(np.float32)
        wav_path = tmp_path / "vio.wav"
        sf.write(str(wav_path), fake_wav, 48000)
        mock_get_path.return_value = str(wav_path)

        data = np.random.default_rng(42).random(100)
        audio, sr = profile_to_wave(data, sr=48000, duration=1, instrument="violin")
        assert len(audio) == 48000

    def test_saves_to_file(self, tmp_path):
        data = np.random.default_rng(42).random(100)
        out = tmp_path / "out.wav"
        audio, sr = profile_to_wave(data, sr=48000, duration=1, output=str(out))
        assert out.exists()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_profile.py -v`
Expected: FAIL

**Step 3: Implement profile.py**

Adapted from `1-profile2wave.py`. Key changes:
- Uses `core.to_profile()` for dimension handling
- Loads instrument WAV via `hub.get_instrument_path()`
- Returns `(audio, sr)` tuple

```python
# src/astrosonify/profile.py
"""Method 1: Convert pulse profile to waveform via interpolation."""

import wave as wave_module
import numpy as np
from scipy.io.wavfile import write as wav_write
from scipy import stats, signal, interpolate

from .core import to_profile, normalize, save_audio
from .hub import get_instrument_path


def _read_wave(file: str) -> np.ndarray:
    """Read a WAV file and return the first channel as int16 array."""
    with wave_module.open(file, "rb") as f:
        nchannels, sampwidth, framerate, nframes = f.getparams()[:4]
        str_data = f.readframes(nframes)
    wave_data = np.frombuffer(str_data, dtype=np.short)
    wave_data = wave_data.reshape(-1, nchannels)
    return wave_data[:, 0]


def profile_to_wave(
    data: np.ndarray,
    sr: int = 48000,
    duration: float = 10.0,
    repeat: int = 10,
    time_downsample: int | None = None,
    instrument: str | None = "violin",
    output: str | None = None,
) -> tuple[np.ndarray, int]:
    """Convert pulse profile to audible waveform.

    If data is 2D (time x freq), it is averaged along the frequency axis.
    The profile is repeated, interpolated to the target duration, and
    optionally convolved with an instrument sample.

    Args:
        data: 1D profile or 2D spectrogram (time x freq).
        sr: Sample rate in Hz.
        duration: Output audio duration in seconds.
        repeat: Number of times to tile the profile.
        time_downsample: Downsample factor for the profile. None = no downsampling.
        instrument: Instrument for convolution ('violin', 'piano', or None).
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    profile = to_profile(data, downsample=time_downsample)
    profile = np.tile(profile, repeat)

    # Interpolate to target sample count
    n_samples = int(sr * duration)
    time_axis = np.arange(len(profile)) / (len(profile) - 1) * duration
    f_interp = interpolate.interp1d(time_axis, profile, kind="linear")
    wave_time = np.linspace(0, duration, n_samples, endpoint=False)
    # Clip to interpolation domain
    wave_time = np.clip(wave_time, time_axis[0], time_axis[-1])
    wave_raw = f_interp(wave_time)

    # Scale to int16 range
    wave_raw = normalize(wave_raw) * 60000
    wave_raw = wave_raw.astype(np.int16)

    if instrument is not None:
        instrument_path = get_instrument_path(instrument)
        sound = _read_wave(instrument_path)

        # Resample instrument to match target sr (original assumed 44100 -> sr)
        n_resampled = int(len(sound) * sr / 44100)
        if n_resampled > 0:
            sounds = stats.binned_statistic(
                x=np.arange(len(sound)),
                values=sound.astype(np.float64),
                bins=n_resampled,
            )[0]
            # Normalize instrument
            integral = np.trapz(sounds)
            if abs(integral) > 1e-10:
                sound_norm = sounds / integral
            else:
                sound_norm = sounds
            # Convolve
            wave_raw = signal.convolve(wave_raw.astype(np.float64), sound_norm, mode="same")
            wave_raw = normalize(wave_raw) * 30000
            wave_raw = wave_raw.astype(np.int16)

    audio = wave_raw.astype(np.float32) / 32768.0

    if output is not None:
        save_audio(audio, sr, output)

    return audio, sr
```

**Step 4: Run tests**

Run: `pytest tests/test_profile.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/astrosonify/profile.py tests/test_profile.py
git commit -m "feat: implement profile_to_wave (method 1)"
```

---

### Task 5: Implement amplitude.py - Method 2 (amplitude modulated sine wave)

**Files:**
- Create: `src/astrosonify/amplitude.py`
- Create: `tests/test_amplitude.py`

**Step 1: Write failing tests**

```python
# tests/test_amplitude.py
import numpy as np
import pytest
from astrosonify.amplitude import amplitude_modulate


class TestAmplitudeModulate:
    def test_returns_tuple(self):
        data = np.random.default_rng(42).random(100)
        audio, sr = amplitude_modulate(data, sr=48000, duration=1)
        assert isinstance(audio, np.ndarray)
        assert sr == 48000

    def test_output_length(self):
        data = np.random.default_rng(42).random(100)
        audio, sr = amplitude_modulate(data, sr=48000, duration=2)
        assert len(audio) == 48000 * 2

    def test_2d_input(self):
        data = np.random.default_rng(42).random((100, 50))
        audio, sr = amplitude_modulate(data, sr=48000, duration=1)
        assert audio.ndim == 1

    def test_saves_to_file(self, tmp_path):
        data = np.random.default_rng(42).random(100)
        out = tmp_path / "out.wav"
        audio, sr = amplitude_modulate(data, sr=48000, duration=1, output=str(out))
        assert out.exists()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_amplitude.py -v`
Expected: FAIL

**Step 3: Implement amplitude.py**

Adapted from `2-amp2loud.py`.

```python
# src/astrosonify/amplitude.py
"""Method 2: Amplitude-modulated sine wave sonification."""

import numpy as np
from scipy.interpolate import interp1d

from .core import to_profile, save_audio


def amplitude_modulate(
    data: np.ndarray,
    sr: int = 48000,
    duration: float = 2.0,
    freq: float = 1000.0,
    time_downsample: int | None = None,
    output: str | None = None,
) -> tuple[np.ndarray, int]:
    """Map pulse profile amplitude to loudness of a sine wave.

    Args:
        data: 1D profile or 2D spectrogram (time x freq).
        sr: Sample rate in Hz.
        duration: Output audio duration in seconds.
        freq: Carrier sine wave frequency in Hz.
        time_downsample: Downsample factor for profile. None = no downsampling.
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    profile = to_profile(data, downsample=time_downsample)
    profile = np.log10(
        (profile - profile.min()) / (profile.max() - profile.min() + 1e-10) + 1
    )

    n_samples = int(sr * duration)
    x_orig = np.linspace(0, 2 * np.pi, len(profile))
    f_interp = interp1d(x_orig, profile)
    x_new = np.linspace(x_orig[0], x_orig[-1], n_samples)
    envelope = f_interp(x_new)

    # Generate AM sine wave: a*sin(freq*x) + b where a=b=envelope
    carrier = np.sin(freq * x_new)
    audio = envelope * carrier + envelope

    # Normalize to [-1, 1]
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    audio = audio.astype(np.float32)

    if output is not None:
        save_audio(audio, sr, output)

    return audio, sr
```

**Step 4: Run tests**

Run: `pytest tests/test_amplitude.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/astrosonify/amplitude.py tests/test_amplitude.py
git commit -m "feat: implement amplitude_modulate (method 2)"
```

---

### Task 6: Implement griffinlim.py - Method 3 (Griffin-Lim vocoder)

**Files:**
- Create: `src/astrosonify/griffinlim.py`
- Create: `tests/test_griffinlim.py`

**Step 1: Write failing tests**

```python
# tests/test_griffinlim.py
import numpy as np
import pytest
from astrosonify.griffinlim import griffinlim


class TestGriffinLim:
    def test_returns_tuple(self):
        rng = np.random.default_rng(42)
        spec = rng.random((64, 128))
        audio, sr = griffinlim(spec, sr=48000, n_iter=10, n_mels=128)
        assert isinstance(audio, np.ndarray)
        assert sr == 48000

    def test_output_is_1d(self):
        rng = np.random.default_rng(42)
        spec = rng.random((64, 128))
        audio, sr = griffinlim(spec, sr=48000, n_iter=10, n_mels=128)
        assert audio.ndim == 1

    def test_rejects_1d(self):
        with pytest.raises(ValueError):
            griffinlim(np.ones(100), sr=48000)

    def test_auto_rebin_freq(self):
        rng = np.random.default_rng(42)
        spec = rng.random((64, 256))
        # Should rebin freq axis to n_mels=128
        audio, sr = griffinlim(spec, sr=48000, n_iter=10, n_mels=128)
        assert audio.ndim == 1

    def test_saves_to_file(self, tmp_path):
        rng = np.random.default_rng(42)
        spec = rng.random((32, 64))
        out = tmp_path / "out.wav"
        audio, sr = griffinlim(spec, sr=48000, n_iter=10, n_mels=64, output=str(out))
        assert out.exists()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_griffinlim.py -v`
Expected: FAIL

**Step 3: Implement griffinlim.py**

Adapted from `3-griffinlim.py`. Key changes:
- Uses `core.rebin_spectrogram()` for dimension adaptation
- Encapsulates all signal processing parameters
- Returns `(audio, sr)` tuple

```python
# src/astrosonify/griffinlim.py
"""Method 3: Griffin-Lim phase reconstruction vocoder."""

import copy
import numpy as np
import librosa
from scipy import signal as scipy_signal

from .core import rebin_spectrogram, del_burst, save_audio


def _mel_to_linear_matrix(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    m = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))


def _griffin_lim(
    spectrogram: np.ndarray,
    n_iter: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> np.ndarray:
    X_best = copy.deepcopy(spectrogram)
    for _ in range(n_iter):
        X_t = librosa.istft(X_best, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hann")
        est = librosa.stft(X_t, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = librosa.istft(X_best, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hann")
    return np.real(X_t)


def griffinlim(
    spectrogram: np.ndarray,
    sr: int = 48000,
    n_iter: int = 200,
    n_mels: int = 512,
    n_fft: int = 4096,
    frame_length: float = 0.04,
    preemphasis: float = 0.97,
    max_db: float = 100.0,
    ref_db: float = 20.0,
    time_rebin: int | None = None,
    freq_rebin: int | None = None,
    clean: bool = False,
    exposure_cut: int = 25,
    output: str | None = None,
) -> tuple[np.ndarray, int]:
    """Reconstruct audio from spectrogram using Griffin-Lim algorithm.

    Treats the input 2D array as a mel-spectrogram and reconstructs
    audio by iteratively estimating phase information.

    Args:
        spectrogram: 2D array (time x freq).
        sr: Sample rate in Hz.
        n_iter: Number of Griffin-Lim iterations.
        n_mels: Number of mel filter banks.
        n_fft: FFT size.
        frame_length: Frame length in seconds (determines win/hop length).
        preemphasis: Pre-emphasis coefficient. Set to 0 to disable.
        max_db: Maximum dB for denormalization.
        ref_db: Reference dB for denormalization.
        time_rebin: Rebin time axis to this many bins. None = auto.
        freq_rebin: Rebin freq axis to this many bins. None = auto to n_mels.
        clean: Whether to apply del_burst cleaning first.
        exposure_cut: Exposure cut for del_burst (only if clean=True).
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    if spectrogram.ndim != 2:
        raise ValueError("griffinlim requires 2D spectrogram input")

    data = spectrogram.astype(np.float64)

    if clean:
        data = del_burst(data, exposure_cut=exposure_cut)

    # Rebin to target dimensions
    target_freq = freq_rebin if freq_rebin is not None else n_mels
    data = rebin_spectrogram(data, time_bins=time_rebin, freq_bins=target_freq)

    # Compute window/hop from frame_length
    win_length = int(sr * frame_length)
    hop_length = win_length // 4

    # Mel spectrogram to waveform
    mel = data.T  # (freq x time)
    mel = (np.clip(mel, 0, 1) * max_db) - max_db + ref_db
    mel = np.power(10.0, mel * 0.05)

    m = _mel_to_linear_matrix(sr, n_fft, data.shape[1])
    mag = np.dot(m, mel)

    wav = _griffin_lim(mag, n_iter, n_fft, hop_length, win_length)

    if preemphasis > 0:
        wav = scipy_signal.lfilter([1], [1, -preemphasis], wav)

    wav, _ = librosa.effects.trim(wav)
    audio = wav.astype(np.float32)

    if output is not None:
        save_audio(audio, sr, output)

    return audio, sr
```

**Step 4: Run tests**

Run: `pytest tests/test_griffinlim.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/astrosonify/griffinlim.py tests/test_griffinlim.py
git commit -m "feat: implement griffinlim vocoder (method 3)"
```

---

### Task 7: Migrate HiFi-GAN model code

**Files:**
- Create: `src/astrosonify/models/hifigan/generator.py` (from `HiFiGAN/models.py` - Generator only)
- Create: `src/astrosonify/models/hifigan/env.py` (from `HiFiGAN/env.py`)
- Modify: `src/astrosonify/models/hifigan/__init__.py`

**Step 1: Copy and adapt env.py**

```python
# src/astrosonify/models/hifigan/env.py
"""HiFi-GAN environment utilities.

Adapted from https://github.com/jik876/hifi-gan (MIT License).
"""


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
```

Note: Only include `AttrDict`, drop `build_env` (training-only utility).

**Step 2: Copy and adapt generator.py**

From `HiFiGAN/models.py`, keep only `Generator`, `ResBlock1`, `ResBlock2`, `get_padding`, `init_weights`. Drop all discriminator classes and loss functions (training-only).

```python
# src/astrosonify/models/hifigan/generator.py
"""HiFi-GAN Generator for mel-spectrogram to waveform conversion.

Adapted from https://github.com/jik876/hifi-gan (MIT License).
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
```

**Step 3: Update __init__.py**

```python
# src/astrosonify/models/hifigan/__init__.py
"""HiFi-GAN neural vocoder model definitions."""
```

**Step 4: Commit**

```bash
git add src/astrosonify/models/hifigan/
git commit -m "feat: migrate HiFi-GAN generator model code (inference only)"
```

---

### Task 8: Implement hifigan.py - Method 4 (HiFi-GAN neural vocoder)

**Files:**
- Create: `src/astrosonify/hifigan.py`

**Step 1: Implement hifigan.py**

Adapted from `4-hifi-gan.py`. Key changes:
- Auto-downloads model from HF Hub
- Uses `core.rebin_spectrogram()` + `skimage.transform.resize` for freq axis
- Returns `(audio, sr)` tuple

```python
# src/astrosonify/hifigan.py
"""Method 4: HiFi-GAN neural vocoder sonification."""

import json
import numpy as np

from .core import rebin_spectrogram, del_burst, normalize, save_audio
from .hub import get_model_path


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "HiFi-GAN method requires PyTorch. "
            "Install with: pip install astrosonify[hifigan]"
        )


def _require_skimage():
    try:
        from skimage.transform import resize
        return resize
    except ImportError:
        raise ImportError(
            "HiFi-GAN method requires scikit-image. "
            "Install with: pip install astrosonify[hifigan]"
        )


def _rescale_data(data: np.ndarray, resize_fn) -> np.ndarray:
    """Rescale spectrogram to 80 mel bins with HiFi-GAN normalization."""
    data = resize_fn(data, (data.shape[0], 80))
    data = normalize(data)
    h, w = data.shape
    a = np.histogram(data.flatten(), bins=int(h * w / 100))
    b, c = (a[1][1:] + a[1][:-1]) / 2, a[0]
    d = 0.6 - b[np.argmax(c)]
    data = (data + d) * 12 - 10.5
    data = np.clip(data, -11, 1.6)
    return data.T[np.newaxis, :, :]  # (1, 80, T)


def hifigan(
    spectrogram: np.ndarray,
    time_rebin: int | None = None,
    clean: bool = False,
    exposure_cut: int = 25,
    output: str | None = None,
) -> tuple[np.ndarray, int]:
    """Convert spectrogram to audio using HiFi-GAN neural vocoder.

    Requires: pip install astrosonify[hifigan]

    The spectrogram frequency axis is automatically resized to 80 mel bins.
    Model weights are downloaded from Hugging Face Hub on first use.

    Args:
        spectrogram: 2D array (time x freq).
        time_rebin: Rebin time axis. None = keep original.
        clean: Whether to apply del_burst cleaning.
        exposure_cut: Exposure cut for del_burst.
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    torch = _require_torch()
    resize_fn = _require_skimage()

    if spectrogram.ndim != 2:
        raise ValueError("hifigan requires 2D spectrogram input")

    data = spectrogram.astype(np.float64)

    if clean:
        data = del_burst(data, exposure_cut=exposure_cut)

    if time_rebin is not None:
        data = rebin_spectrogram(data, time_bins=time_rebin)

    # Download model files
    config_path = get_model_path("hifigan", "config.json")
    checkpoint_path = get_model_path("hifigan", "generator.pth")

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    from .models.hifigan.env import AttrDict
    from .models.hifigan.generator import Generator

    h = AttrDict(config)
    torch.manual_seed(h.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)

    generator = Generator(h).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict["generator"])
    generator.eval()
    generator.remove_weight_norm()

    # Rescale input to (1, 80, T)
    x = _rescale_data(data, resize_fn)

    with torch.no_grad():
        x_tensor = torch.FloatTensor(x).to(device)
        audio_tensor = generator(x_tensor).squeeze()
        audio = audio_tensor.cpu().numpy()

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    audio = audio.astype(np.float32)

    sr = h.sampling_rate

    if output is not None:
        save_audio(audio, sr, output)

    return audio, sr
```

**Step 2: Commit**

```bash
git add src/astrosonify/hifigan.py
git commit -m "feat: implement hifigan neural vocoder (method 4)"
```

Note: Full testing of HiFi-GAN requires torch + model weights. Skipped in basic test suite.

---

### Task 9: Migrate MusicNet model code

**Files:**
- Create: `src/astrosonify/models/musicnet/wavenet.py` (from `MusicNet/wavenet.py`)
- Create: `src/astrosonify/models/musicnet/wavenet_models.py` (from `MusicNet/wavenet_models.py`)
- Create: `src/astrosonify/models/musicnet/wavenet_generator.py` (from `MusicNet/wavenet_generator.py`)
- Create: `src/astrosonify/models/musicnet/utils.py` (from `MusicNet/utils.py` - only mu_law/inv_mu_law/timeit)
- Modify: `src/astrosonify/models/musicnet/__init__.py`

**Step 1: Copy model files**

Copy the four files from `MusicNet/` to `src/astrosonify/models/musicnet/`, preserving original license headers. Key modifications:
- Fix internal imports to use relative imports (e.g., `from .utils import ...`, `from .wavenet import ...`)
- In `utils.py`: Keep only `mu_law`, `inv_mu_law`, `timeit`. Drop logging/plotting utilities.
- Preserve Facebook copyright headers.

**Step 2: Update __init__.py**

```python
# src/astrosonify/models/musicnet/__init__.py
"""MusicNet WaveNet model definitions.

Adapted from https://github.com/facebookresearch/music-translation
Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
```

**Step 3: Commit**

```bash
git add src/astrosonify/models/musicnet/
git commit -m "feat: migrate MusicNet WaveNet model code (inference only)"
```

---

### Task 10: Implement musicnet.py - Method 5 (WaveNet style transfer)

**Files:**
- Create: `src/astrosonify/musicnet.py`

**Step 1: Implement musicnet.py**

Adapted from `5-musicnet.py`. Key changes:
- Auto-downloads model from HF Hub
- Accepts WAV path or 1D audio array
- Handles Windows PosixPath issue internally
- Returns `(audio, sr)` tuple

```python
# src/astrosonify/musicnet.py
"""Method 5: WaveNet-based music style transfer."""

import pathlib
import platform
import numpy as np
from scipy.io import wavfile

from .core import save_audio
from .hub import get_model_path


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "MusicNet method requires PyTorch. "
            "Install with: pip install astrosonify[musicnet]"
        )


def _require_tqdm():
    try:
        import tqdm
        return tqdm
    except ImportError:
        raise ImportError(
            "MusicNet method requires tqdm. "
            "Install with: pip install astrosonify[musicnet]"
        )


STYLE_NAMES = {
    0: "Accompaniment Violin (Beethoven)",
    1: "Solo Cello (Bach)",
    2: "Solo Piano (Bach)",
    3: "Solo Piano (Beethoven)",
    4: "String Quartet (Beethoven)",
    5: "Organ Quintet (Cambini)",
}


def musicnet(
    input_audio,
    decoder_id: int = 2,
    checkpoint_type: str = "bestmodel",
    sr: int = 48000,
    batch_size: int = 1,
    split_size: int = 20,
    output: str | None = None,
) -> tuple[np.ndarray, int]:
    """Apply music style transfer using WaveNet encoder-decoder.

    Requires: pip install astrosonify[musicnet]
    Requires CUDA GPU for inference.

    Args:
        input_audio: Path to WAV file, or 1D numpy audio array.
        decoder_id: Style decoder ID (0-5). See STYLE_NAMES for mapping.
        checkpoint_type: 'bestmodel' or 'lastmodel'.
        sr: Sample rate for loading/output.
        batch_size: Batch size for inference.
        split_size: Split size for autoregressive generation.
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    torch = _require_torch()
    tqdm_mod = _require_tqdm()
    import librosa

    if decoder_id not in range(6):
        raise ValueError(f"decoder_id must be 0-5. Available styles: {STYLE_NAMES}")
    if checkpoint_type not in ("bestmodel", "lastmodel"):
        raise ValueError("checkpoint_type must be 'bestmodel' or 'lastmodel'")

    # Handle Windows PosixPath issue with torch.load
    posix_backup = None
    if platform.system() == "Windows":
        posix_backup = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath

    try:
        # Download model files
        checkpoint_file = f"{checkpoint_type}_{decoder_id}.pth"
        checkpoint_path = get_model_path("musicnet", checkpoint_file)
        args_path = get_model_path("musicnet", "args.pth")

        from .models.musicnet import wavenet_models, utils
        from .models.musicnet.wavenet import WaveNet
        from .models.musicnet.wavenet_generator import WavenetGenerator

        model_args = torch.load(args_path, map_location="cpu")[0]

        encoder = wavenet_models.Encoder(model_args)
        state = torch.load(checkpoint_path, map_location="cpu")
        encoder.load_state_dict(state["encoder_state"])
        encoder.eval()
        encoder = encoder.cuda()

        decoder = WaveNet(model_args)
        decoder.load_state_dict(state["decoder_state"])
        decoder.eval()
        decoder = decoder.cuda()
        decoder = WavenetGenerator(decoder, batch_size=batch_size, wav_freq=sr)

        # Load audio
        if isinstance(input_audio, (str, pathlib.Path)):
            data, _ = librosa.load(str(input_audio), sr=sr)
        else:
            data = np.asarray(input_audio, dtype=np.float32)

        data = utils.mu_law(data)
        xs = torch.stack([torch.tensor(data).unsqueeze(0).float().cuda()]).contiguous()

        with torch.no_grad():
            zz = torch.cat([encoder(xs_batch) for xs_batch in torch.split(xs, batch_size)], dim=0)
            audio_res = []
            for zz_batch in torch.split(zz, batch_size):
                splits = torch.split(zz_batch, split_size, -1)
                audio_data = []
                decoder.reset()
                for cond in tqdm_mod.tqdm(splits, desc="Generating"):
                    audio_data.append(decoder.generate(cond).cpu())
                audio_data = torch.cat(audio_data, -1)
                audio_res.append(audio_data)
            audio_res = torch.cat(audio_res, dim=0)

        audio = utils.inv_mu_law(audio_res.cpu().numpy()).squeeze()
        audio = audio.astype(np.float32)

    finally:
        if posix_backup is not None:
            pathlib.PosixPath = posix_backup

    if output is not None:
        save_audio(audio, sr, output)

    return audio, sr
```

**Step 2: Commit**

```bash
git add src/astrosonify/musicnet.py
git commit -m "feat: implement musicnet style transfer (method 5)"
```

---

### Task 11: Implement astronify_method.py - Method 0

**Files:**
- Create: `src/astrosonify/astronify_method.py`

**Step 1: Implement astronify_method.py**

Adapted from `0-astronify.py`.

```python
# src/astrosonify/astronify_method.py
"""Method 0: Sonification using the astronify library."""

import numpy as np

from .core import to_profile, save_audio


def _require_astronify():
    try:
        from astropy.table import Table
        from astronify.series import SoniSeries
        return Table, SoniSeries
    except ImportError:
        raise ImportError(
            "Astronify method requires astropy and astronify. "
            "Install with: pip install astrosonify[astronify]"
        )


def astronify_sonify(
    data: np.ndarray,
    note_spacing: float = 0.01,
    time_downsample: int = 10,
    output: str | None = None,
) -> tuple[np.ndarray, int]:
    """Sonify data using the astronify library.

    Maps pulse profile intensity to pitch.

    Requires: pip install astrosonify[astronify]

    Args:
        data: 1D profile or 2D spectrogram (time x freq).
        note_spacing: Time between notes in seconds.
        time_downsample: Downsample factor for the profile.
        output: Path to save WAV file. None = don't save.

    Returns:
        Tuple of (audio_array, sample_rate). Note: astronify generates
        audio internally; the returned array is read back from the
        written file if output is provided, otherwise a temporary file
        is used.
    """
    Table, SoniSeries = _require_astronify()

    profile = to_profile(data, downsample=time_downsample)
    profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-10)

    data_table = Table({"time": np.arange(len(profile)), "flux": profile})
    soni = SoniSeries(data_table)
    soni.note_spacing = note_spacing
    soni.sonify()

    if output is not None:
        soni.write(output)

    # Extract audio data from astronify's internal representation
    import tempfile, os, soundfile as sf
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        soni.write(tmp_path)
        audio, sr = sf.read(tmp_path)
    finally:
        os.unlink(tmp_path)

    return audio.astype(np.float32), sr
```

**Step 2: Commit**

```bash
git add src/astrosonify/astronify_method.py
git commit -m "feat: implement astronify_sonify (method 0)"
```

---

### Task 12: Wire up __init__.py with public API exports

**Files:**
- Modify: `src/astrosonify/__init__.py`

**Step 1: Update __init__.py**

```python
# src/astrosonify/__init__.py
"""AstroSonify - Methods for Sonifying Pulse.

Convert radio telescope time-frequency data into audible sound
using multiple sonification techniques.
"""

__version__ = "0.1.0"

from .core import (
    normalize,
    del_burst,
    rebin_spectrogram,
    to_profile,
    save_audio,
)
from .hub import load_example
from .profile import profile_to_wave
from .amplitude import amplitude_modulate
from .griffinlim import griffinlim

# Lazy imports for optional-dependency methods
def astronify_sonify(*args, **kwargs):
    """Sonify using astronify library. Requires: pip install astrosonify[astronify]"""
    from .astronify_method import astronify_sonify as _impl
    return _impl(*args, **kwargs)

def hifigan(*args, **kwargs):
    """HiFi-GAN neural vocoder. Requires: pip install astrosonify[hifigan]"""
    from .hifigan import hifigan as _impl
    return _impl(*args, **kwargs)

def musicnet(*args, **kwargs):
    """WaveNet style transfer. Requires: pip install astrosonify[musicnet]"""
    from .musicnet import musicnet as _impl
    return _impl(*args, **kwargs)

__all__ = [
    "__version__",
    "normalize",
    "del_burst",
    "rebin_spectrogram",
    "to_profile",
    "save_audio",
    "load_example",
    "astronify_sonify",
    "profile_to_wave",
    "amplitude_modulate",
    "griffinlim",
    "hifigan",
    "musicnet",
]
```

**Step 2: Verify import works**

Run: `python -c "import astrosonify; print(dir(astrosonify))"`
Expected: Shows all exported names without ImportError

**Step 3: Commit**

```bash
git add src/astrosonify/__init__.py
git commit -m "feat: wire up public API exports with lazy imports for optional deps"
```

---

### Task 13: Implement CLI

**Files:**
- Create: `src/astrosonify/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write failing tests**

```python
# tests/test_cli.py
import numpy as np
import pytest
from click.testing import CliRunner
from astrosonify.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def npy_file(tmp_path):
    data = np.random.default_rng(42).random((64, 128))
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
        assert result.exit_code == 0

    def test_amplitude_command(self, runner, profile_file, tmp_path):
        out = str(tmp_path / "out.wav")
        result = runner.invoke(main, [
            "amplitude", "--input", profile_file, "--output", out,
            "--duration", "0.5"
        ])
        assert result.exit_code == 0

    def test_griffinlim_command(self, runner, npy_file, tmp_path):
        out = str(tmp_path / "out.wav")
        result = runner.invoke(main, [
            "griffinlim", "--input", npy_file, "--output", out,
            "--n-iter", "5"
        ])
        assert result.exit_code == 0

    def test_missing_input(self, runner, tmp_path):
        out = str(tmp_path / "out.wav")
        result = runner.invoke(main, ["griffinlim", "--output", out])
        assert result.exit_code != 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL

**Step 3: Implement cli.py**

```python
# src/astrosonify/cli.py
"""AstroSonify command-line interface."""

import click
import numpy as np


@click.group()
@click.version_option()
def main():
    """AstroSonify - Convert radio telescope data into audible sound."""
    pass


@main.command()
def list_methods():
    """List available sonification methods."""
    methods = [
        ("astronify", "Map profile intensity to pitch (requires astronify)"),
        ("profile", "Convert pulse profile to waveform with instrument convolution"),
        ("amplitude", "Amplitude-modulated sine wave"),
        ("griffinlim", "Griffin-Lim phase reconstruction vocoder"),
        ("hifigan", "HiFi-GAN neural vocoder (requires torch)"),
        ("musicnet", "WaveNet music style transfer (requires torch + CUDA)"),
    ]
    for name, desc in methods:
        click.echo(f"  {name:12s}  {desc}")


@main.command()
@click.option("--input", "input_path", required=True, help="Input .npy file")
@click.option("--output", "output_path", required=True, help="Output .wav file")
@click.option("--sr", default=48000, help="Sample rate (Hz)")
@click.option("--duration", default=10.0, help="Duration (seconds)")
@click.option("--repeat", default=10, help="Profile repeat count")
@click.option("--instrument", default="violin", help="Instrument (violin/piano)")
@click.option("--no-instrument", is_flag=True, help="Disable instrument convolution")
@click.option("--downsample", default=None, type=int, help="Time downsample factor")
def profile(input_path, output_path, sr, duration, repeat, instrument, no_instrument, downsample):
    """Sonify using pulse profile to waveform (Method 1)."""
    from .profile import profile_to_wave
    data = np.load(input_path)
    inst = None if no_instrument else instrument
    profile_to_wave(data, sr=sr, duration=duration, repeat=repeat,
                    instrument=inst, time_downsample=downsample, output=output_path)
    click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, help="Input .npy file")
@click.option("--output", "output_path", required=True, help="Output .wav file")
@click.option("--sr", default=48000, help="Sample rate (Hz)")
@click.option("--duration", default=2.0, help="Duration (seconds)")
@click.option("--freq", default=1000.0, help="Carrier frequency (Hz)")
@click.option("--downsample", default=None, type=int, help="Time downsample factor")
def amplitude(input_path, output_path, sr, duration, freq, downsample):
    """Sonify using amplitude modulation (Method 2)."""
    from .amplitude import amplitude_modulate
    data = np.load(input_path)
    amplitude_modulate(data, sr=sr, duration=duration, freq=freq,
                       time_downsample=downsample, output=output_path)
    click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, help="Input .npy file")
@click.option("--output", "output_path", required=True, help="Output .wav file")
@click.option("--sr", default=48000, help="Sample rate (Hz)")
@click.option("--n-iter", default=200, help="Griffin-Lim iterations")
@click.option("--n-mels", default=512, help="Number of mel bands")
@click.option("--n-fft", default=4096, help="FFT size")
@click.option("--clean", is_flag=True, help="Apply burst cleaning")
def griffinlim(input_path, output_path, sr, n_iter, n_mels, n_fft, clean):
    """Sonify using Griffin-Lim vocoder (Method 3)."""
    from .griffinlim import griffinlim as gl
    data = np.load(input_path)
    gl(data, sr=sr, n_iter=n_iter, n_mels=n_mels, n_fft=n_fft,
       clean=clean, output=output_path)
    click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, help="Input .npy file")
@click.option("--output", "output_path", required=True, help="Output .wav file")
@click.option("--clean", is_flag=True, help="Apply burst cleaning")
def hifigan(input_path, output_path, clean):
    """Sonify using HiFi-GAN neural vocoder (Method 4)."""
    from .hifigan import hifigan as hf
    data = np.load(input_path)
    hf(data, clean=clean, output=output_path)
    click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, help="Input .wav file")
@click.option("--output", "output_path", required=True, help="Output .wav file")
@click.option("--decoder-id", default=2, type=int, help="Style decoder (0-5)")
@click.option("--checkpoint-type", default="bestmodel", help="bestmodel or lastmodel")
@click.option("--sr", default=48000, help="Sample rate (Hz)")
def musicnet(input_path, output_path, decoder_id, checkpoint_type, sr):
    """Sonify using WaveNet style transfer (Method 5)."""
    from .musicnet import musicnet as mn
    mn(input_path, decoder_id=decoder_id, checkpoint_type=checkpoint_type,
       sr=sr, output=output_path)
    click.echo(f"Saved to {output_path}")


@main.command("download-examples")
@click.option("--dest", default="./data", help="Destination directory")
def download_examples(dest):
    """Download example data files from Hugging Face Hub."""
    import os
    from .hub import load_example, EXAMPLE_MAP
    os.makedirs(dest, exist_ok=True)
    for name, filename in EXAMPLE_MAP.items():
        click.echo(f"Downloading {name} ({filename})...")
        data = load_example(name)
        out_path = os.path.join(dest, filename)
        np.save(out_path, data)
        click.echo(f"  Saved to {out_path}")
    click.echo("Done!")
```

**Step 4: Run tests**

Run: `pytest tests/test_cli.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/astrosonify/cli.py tests/test_cli.py
git commit -m "feat: implement CLI with click (all methods + download-examples)"
```

---

### Task 14: Update .gitignore and clean up

**Files:**
- Modify: `.gitignore`

**Step 1: Update .gitignore**

Add these entries to `.gitignore`:

```
# Large data files (now hosted on HF Hub)
Data/*.npy
Data/*.wav

# Large model files (now hosted on HF Hub)
HiFiGAN/model/
MusicNet/checkpoints/pretrained_musicnet/

# Build artifacts
dist/
build/
*.egg-info/
src/*.egg-info/
```

**Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: update .gitignore for package build and large files"
```

---

### Task 15: Run full test suite and verify package builds

**Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 2: Verify package builds**

Run: `pip install build && python -m build`
Expected: Creates `dist/astrosonify-0.1.0.tar.gz` and `dist/astrosonify-0.1.0-py3-none-any.whl`

**Step 3: Verify the wheel installs in a clean environment**

Run:
```bash
python -m venv /tmp/test_env
source /tmp/test_env/bin/activate
pip install dist/astrosonify-0.1.0-py3-none-any.whl
python -c "import astrosonify; print(astrosonify.__version__)"
astrosonify --help
astrosonify list-methods
deactivate
```
Expected: All commands succeed

**Step 4: Commit any fixes**

If any issues arise, fix and commit.

---

### Task 16: Final commit and tag

**Step 1: Create version tag**

```bash
git tag -a v0.1.0 -m "Initial release: astrosonify v0.1.0"
```

**Step 2: Summary**

Package is ready for:
1. Upload to PyPI: `twine upload dist/*`
2. Upload models/data to Hugging Face Hub (`SukiYume/astrosonify`)
3. Users install with: `pip install astrosonify`
