# RadioSonify

[中文说明](README_CN.md)

RadioSonify converts one-, two-, and three-dimensional numerical data into duration-controlled audio. A single Python API and command-line workflow cover scientific preprocessing, sonification, optional neural timbre conversion, WAV output, and reproducibility metadata.

## Supported data

| Input | Canonical layout | Default method | Audio |
|---|---|---|---|
| 1-D profile | `(time,)` | `amplitude` | mono |
| 2-D matrix | `(time, feature)` | `erb` | mono |
| 3-D layered matrix | `(layer, time, feature)` | `spatial_erb` | stereo |

Two-dimensional inputs can represent dynamic spectra, spectrograms, images, or any ordered time-by-feature matrix. Three-dimensional inputs can represent polarization products, image channels, sensor layers, or other parallel matrices. `data_duration` gives the physical time span represented by the canonical time axis.

## Installation

RadioSonify supports Python 3.9 through 3.13.

```bash
git clone https://github.com/SukiYume/MSP.git
cd MSP
python -m pip install .
```

Install the extras used by each neural backend:

```bash
python -m pip install ".[hifigan]"
python -m pip install ".[musicnet]"
python -m pip install ".[rave]"
python -m pip install ".[all]"
```

## First sonification

```python
from pathlib import Path

import radiosonify as rs

data = rs.load_example("raw_burst")
result = rs.sonify(
    data,
    data_duration=2.4,
    method="erb",
    preprocess_params={"scale_statistic": "mad"},
    output=Path("output/burst.wav"),
)

print(result.sample_rate, result.output_duration)
print(result.preprocess_params)
print(result.method_params)
```

The equivalent command is:

```bash
radiosonify download-examples --dest ./data
radiosonify sonify \
  --input data/RawBurst.npy \
  --output output/burst.wav \
  --duration 2.4 \
  --method erb \
  --preprocess scale_statistic=mad
```

CLI settings use repeatable `KEY=VALUE` options. Python literal syntax handles numbers, tuples, dictionaries, booleans, and `None`; plain words become strings.

## Input axes and snapshots

`SonificationInput` stores a canonical immutable `float64` snapshot. It preserves the caller's original shape and axis declarations in the result.

| Dimensionality | Default source axes |
|---|---|
| 1-D | time axis `0` |
| 2-D | time axis `0` |
| 3-D | layer axis `0`, time axis `1` |

Declare another source ordering explicitly:

```python
import numpy as np

cube = np.load("layers.npy")
source = rs.SonificationInput(
    cube,
    duration=3.0,
    layer_axis=2,
    time_axis=0,
    name="observation-17",
)
result = rs.sonify(source, method="spatial_erb")
```

The standard input domain contains real finite values. `nan_policy="propagate"` treats NaN values as masks and maps them to silence after normalization. Complex values and infinities produce validation errors.

## Processing sequence

Every call resolves one immutable execution plan before array transformation. This planning stage validates the selected method, all settings, feature and frame geometry, planned layer count, output path, optional dependencies, model assets, and neural channel contracts. Execution then follows this sequence:

```text
canonical input snapshot
→ layer/time/feature resizing
→ baseline and scale calibration
→ optional percentile clipping
→ timeline repetition and smoothing
→ normalization to [0, 1]
→ primary sonification
→ duration fitting
→ optional audio postprocessor
→ output sample-rate conversion and WAV conditioning
```

The separation gives each stage one data contract: preprocessing owns scientific-array transformations, primary methods own audible mapping, postprocessors own audio-domain style conversion, and output conditioning owns the final container.

## Shared preprocessing

| Setting | Purpose |
|---|---|
| `layer_rebin` | target layer count for 3-D data, using ordered area averaging |
| `time_rebin` | target time-bin count; methods with frame geometry can resolve `"auto"` |
| `feature_rebin` | target feature-bin count |
| `baseline_operation` | `"subtract"`, `"divide"`, or `None` |
| `baseline_statistic` | `"median"` or `"mean"` |
| `baseline_axis` | calibration axis or `"auto"` |
| `scale_statistic` | per-channel `"mad"`, `"std"`, or `None` |
| `clip_percentiles` | global `(lower, upper)` percentile pair or `None` |
| `time_smoothing` | Gaussian sigma along the canonical time axis or `None` |
| `normalization_scope` | `"global"`, `"per_layer"`, or `"auto"` |
| `nan_policy` | `"raise"` or `"propagate"` |

Downsampling uses equal-width area averages across the full source extent. Upsampling on time and feature axes uses bin-center interpolation. `layer_rebin` performs dimensional reduction and retains layer order. Three-dimensional data defaults to per-layer normalization so each parallel layer remains audible; `layer_gains` carries explicit scientific weighting into spatial synthesis.

`repeat` joins copies along the canonical time axis before temporal smoothing. This places repeated observations on one continuous timeline.

The preprocessing stage is also available for analysis and inspection:

```python
import numpy as np

layers = np.load("layers.npy")
prepared_layers = rs.preprocess(
    layers,
    layer_rebin=4,
    time_rebin=2048,
    feature_rebin=512,
    scale_statistic="mad",
)
defaults = rs.preprocessing_defaults()
```

## Primary methods

| Method | Inputs | Mapping | Main controls | Extra |
|---|---|---|---|---|
| `profile` | 1-D, 2-D | interpolated profile waveform with an optional analytic instrument response | `sr`, `instrument` | base |
| `amplitude` | 1-D, 2-D | profile amplitude envelope on a harmonic carrier | `sr`, `freq`, `compression`, `harmonics`, `harmonic_decay` | base |
| `erb` | 2-D | time to time, ordered feature position to perceptual pitch, brightness and temporal salience to level | frequency range, band count, timbre, envelope, event, and level settings | base |
| `griffinlim` | 2-D | mel-like magnitude interpretation with deterministic iterative phase reconstruction | `sr`, `n_iter`, `n_fft`, `frame_length`, `preemphasis`, `max_db`, `ref_db` | base |
| `hifigan` | 2-D | checkpoint-specific log-mel adapter followed by a HiFi-GAN vocoder | registered model geometry | `hifigan` |
| `spatial_erb` | 3-D | one ERB synthesis per layer with constant-power stereo panning | ERB controls, `pan_positions`, `layer_gains` | base |

### ERB and spatial ERB

ERB synthesis uses overlapping perceptual bands, phase-continuous carriers, attack/release smoothing, bounded auditory-level compensation, RMS control, and true-peak limiting. `frequency_scale="mel"` follows an HTK mel spacing; `frequency_scale="erb"` selects ERB-rate spacing. `n_bands` controls spectral detail, and `None` derives the count from the selected frequency range.

Available timbres are `sine`, `retro_digital`, `warm_pad`, `soft_marimba`, `glass_bell`, and `instrument_palette`. `instrument_palette` crossfades complementary voices over pitch. `event_voice="water_drop"` adds deterministic transient accents from temporal salience.

```python
matrix = rs.load_example("raw_burst")
result = rs.sonify(
    matrix,
    data_duration=2.4,
    method="erb",
    method_params={
        "min_freq": 90.0,
        "max_freq": 6000.0,
        "n_bands": 48,
        "frequency_scale": "mel",
        "timbre": "instrument_palette",
        "event_voice": "water_drop",
        "voice_params": {"harmonic_limit_hz": 6500.0},
        "event_params": {"max_events_per_second": 3.0, "level_db": -16.0},
    },
)
```

For `spatial_erb`, `pan_positions` ranges from `-1` to `1` and `layer_gains` contains values of `0` or greater. Each sequence has one value for every planned layer after `layer_rebin`.

### Griffin-Lim and HiFi-GAN

Griffin-Lim derives its valid feature count and automatic time-bin count from `n_fft`, `frame_length`, sample rate, duration, and repetition. HiFi-GAN accepts the shared normalized matrix and applies its published checkpoint's fixed 80-bin encoding within the model adapter. The data-dependent histogram offset appears in `result.method_params`.

## Audio postprocessors

`musicnet` converts mono primary audio into one of six pretrained WaveNet music styles at 16 kHz. Its encoder requires at least 800 samples after resampling, equivalent to 50 ms of primary audio. Planning validates that length and resolves dependencies and pinned model assets before scientific preprocessing.

`rave` applies a user-supplied exported TorchScript model. Planning loads the model on CPU, reads the standard nn~ `sampling_rate` and `forward_params` metadata, resolves input and output sample rates, and verifies channel compatibility before scientific preprocessing. Inference reloads the same contract on the selected `cpu`, `cuda`, or `mps` device. Mono one-in/one-out models process stereo channels independently, and a mono source can expand into a model's multichannel input.

Use RAVE exports from trusted sources because `torch.jit.load` executes model code. Record each model's origin and license with the generated audio.

```python
profile = rs.load_example("profile")
matrix = rs.load_example("raw_burst")
music_style = rs.sonify(
    profile,
    data_duration=2.0,
    method="amplitude",
    postprocess="musicnet",
    postprocess_params={"decoder_id": 2, "seed": 0},
)

rave_style = rs.sonify(
    matrix,
    data_duration=2.0,
    method="erb",
    postprocess="rave",
    postprocess_params={
        "model_path": "/path/to/trusted-model.ts",
        "device": "auto",
        "seed": 0,
    },
)
```

## Duration and output

All methods use one duration formula:

```text
target_duration = data_duration × repeat ÷ speed
```

`speed=2` produces half the source duration, and `speed=0.5` produces twice the source duration. The registered repeat default is `5` for `amplitude` and `1` for every other primary method. An explicit `repeat` overrides the registered value.

`preserve_pitch=True` selects phase-vocoder time stretching. The standard polyphase path changes playback speed and pitch together. `output_sr` selects the final sample rate while preserving duration and physical pitch. The final sample count is `round(output_sr × target_duration)` when `output_sr` is supplied.

Output conditioning removes DC, applies short edge fades, constrains peak level, creates parent directories, and writes WAV data after path validation.

## Result and reproducibility

`sonify` returns an immutable `SonificationResult`.

| Fields | Contents |
|---|---|
| `audio`, `sample_rate`, `output_duration`, `output_path` | final waveform and container |
| `data_type`, `data_duration`, `input_shape`, `source_name` | source identity and physical extent |
| `source_time_axis`, `source_layer_axis` | axes in the caller's original layout |
| `method`, `preprocess_params`, `method_params` | primary mapping and fully resolved settings |
| `speed`, `repeat`, `preserve_pitch`, `target_duration` | timing contract |
| `method_sample_rate`, `method_native_samples`, `method_native_duration`, `method_time_scale` | primary synthesis timing |
| `postprocess`, `postprocess_params`, `postprocess_native_samples`, `postprocess_native_duration`, `postprocess_time_scale` | optional audio-style stage |

Parameter mappings are recursively copied and frozen. Numeric arrays use immutable byte-backed snapshots. A reproducible publication can record the RadioSonify version, source checksum, source axes, physical duration, result parameter mappings, model revision, and output sample rate.

## Discovery and example assets

```bash
radiosonify list-methods
radiosonify list-settings
radiosonify --help
radiosonify download-examples --dest ./data
```

```python
rs.available_methods()
rs.available_methods("matrix")
rs.available_postprocessors()
rs.default_method("matrix")

profile = rs.load_example("profile")
burst = rs.load_example("burst")
raw_burst = rs.load_example("raw_burst")
parkes_burst = rs.load_example("parkes_burst")
```

Example arrays, HiFi-GAN weights, and MusicNet checkpoints come from the pinned [`TorchLight/radiosonify`](https://huggingface.co/TorchLight/radiosonify) revision stored in `radiosonify.hub.REVISION`. Assets download on first use into `~/.cache/radiosonify`; `RADIOSONIFY_CACHE_DIR` selects another cache directory. Analytic `profile` instrument responses are generated locally and cached alongside downloaded assets. [MODEL_ASSETS.md](MODEL_ASSETS.md) records origins, transformations, verification history, and license scope.

## Development and license

[CONTRIBUTING.md](CONTRIBUTING.md) contains the development setup, module boundaries, and validation commands. [CHANGELOG.md](CHANGELOG.md) records each release.

MSP-authored code uses the [MIT License](LICENSE). The vendored MusicNet inference subset and checkpoints use CC BY-NC 4.0 with a non-commercial-use condition. Distribution metadata uses `MIT AND CC-BY-NC-4.0`, and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) gives component-level terms.
