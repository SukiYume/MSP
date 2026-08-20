# RadioSonify

[中文说明](README_CN.md)

RadioSonify converts one-, two-, and three-dimensional numerical arrays into duration-controlled audio. It provides a unified Python API, a command-line interface, deterministic signal-processing methods, optional neural timbre transforms, and provenance for each result.

## Project overview

| Input | Standard layout | Default method | Output |
|---|---|---|---|
| 1-D profile | `(time,)` | `amplitude` | mono |
| 2-D matrix | `(time, feature)` | `erb` | mono |
| 3-D layered matrix | `(layer, time, feature)` | `spatial_erb` | stereo |

The physical span of the input is supplied through `data_duration`. RadioSonify applies shared preprocessing, selects or validates a sonification method, fits the requested duration, conditions the waveform, and optionally writes a WAV file.

## Installation

RadioSonify supports Python 3.9 through 3.13. Install the project from source:

```bash
git clone https://github.com/SukiYume/MSP.git
cd MSP
python -m pip install .
```

Install a neural backend when its method is needed:

```bash
python -m pip install ".[hifigan]"
python -m pip install ".[musicnet]"
python -m pip install ".[rave]"
python -m pip install ".[all]"
```

## Example data and cached assets

Four example arrays are available for a first run:

```python
import radiosonify as rs

profile = rs.load_example("profile")             # 1-D
burst = rs.load_example("burst")                 # 2-D, corrected
raw_burst = rs.load_example("raw_burst")         # 2-D, as recorded
parkes_burst = rs.load_example("parkes_burst")   # 2-D, as recorded
```

The same arrays are available as files:

```bash
radiosonify download-examples --dest ./data
```

Example arrays, HiFi-GAN weights, and MusicNet checkpoints come from a pinned [`TorchLight/radiosonify`](https://huggingface.co/TorchLight/radiosonify) revision recorded in `radiosonify.hub.REVISION`, and each file downloads on first use. Downloads land in `~/.cache/radiosonify`; set `RADIOSONIFY_CACHE_DIR` before importing the package to choose another location. The `profile` method's instrument responses are synthesized locally and cached in the same directory. [MODEL_ASSETS.md](MODEL_ASSETS.md) records asset origins, transformations, verification history, and license scope.

## Quick start

### Python

```python
from pathlib import Path

import numpy as np
import radiosonify as rs

profile_result = rs.sonify(
    rs.load_example("profile"),
    data_duration=1.2,
    method="amplitude",
    repeat=1,
)

matrix_result = rs.sonify(
    rs.load_example("raw_burst"),
    data_duration=2.4,
    method="erb",
    preprocess_params={"scale_statistic": "mad"},
    output=Path("output/matrix.wav"),
)

spatial_result = rs.sonify(
    np.load("layers.npy"),
    data_duration=3.0,
    method="spatial_erb",
)
```

Each result contains the frozen audio snapshot, sample rate, selected method, source geometry, timing values, effective preprocessing settings, method settings, postprocessor settings, and output path.

### Command line

```bash
radiosonify sonify \
  --input matrix.npy \
  --output output/matrix.wav \
  --duration 2.4 \
  --method erb \
  --preprocess scale_statistic=mad
```

CLI setting values use repeatable `KEY=VALUE` options. Numeric values, tuples, booleans, and `None` use Python literal syntax; plain words are parsed as strings.

### Discovery

The installed parameter surface is available from both interfaces:

```bash
radiosonify list-methods
radiosonify list-settings
radiosonify --help
```

```python
rs.available_methods()           # every registered primary method
rs.available_methods("matrix")   # methods accepting one input type
rs.available_postprocessors()
rs.default_method("matrix")
```

## Input contract

`SonificationInput` copies the caller's data into a canonical `float64` array backed by an immutable buffer. Changes to the caller's array leave the stored snapshot unchanged, and the snapshot retains structural write protection under NumPy's write-flag API.

The default axis rules are:

| Dimensionality | Meaning | Default axes |
|---|---|---|
| 1-D | time profile | time axis `0` |
| 2-D | time-by-feature matrix | time axis `0` |
| 3-D | parallel time-by-feature layers | layer axis `0`, time axis `1` after canonicalization |

Declare another source ordering when required:

```python
source = rs.SonificationInput(
    data,
    duration=2.0,
    time_axis=1,
    layer_axis=2,
    name="observation-17",
)
result = rs.sonify(source, method="spatial_erb")
```

Finite real values form the standard input domain. `nan_policy="propagate"` treats NaN values as masked samples and maps them to silence after normalization. Infinite and complex values raise a validation error.

## Shared preprocessing

All primary methods receive arrays normalized to `[0, 1]`. The pipeline runs in this order:

```text
time/feature rebinning
→ baseline correction
→ per-channel scale correction
→ percentile clipping
→ repetition
→ temporal smoothing
→ min-max normalization
```

| Setting | Purpose |
|---|---|
| `time_rebin` | target number of time bins; registered methods may select `"auto"` |
| `feature_rebin` | target number of feature bins |
| `baseline_operation` | `"subtract"`, `"divide"`, or `None` |
| `baseline_statistic` | `"median"` or `"mean"` |
| `baseline_axis` | source axis used by the baseline statistic and by `scale_statistic`, or `"auto"` |
| `scale_statistic` | per-channel `"mad"`, `"std"`, or `None` |
| `clip_percentiles` | `(lower, upper)` percentile pair measured across the whole array, or `None` |
| `time_smoothing` | Gaussian width along time or `None` |
| `normalization_scope` | `"global"`, `"per_layer"`, or `"auto"` |
| `nan_policy` | `"raise"` or `"propagate"` |

`repeat` belongs to the unified timing contract and joins copies along the canonical time axis. Temporal smoothing spans the joined boundaries.

The stage also runs on its own, which is what each method-specific CLI command does before dispatch:

```python
normalized = rs.preprocess(raw, scale_statistic="mad", time_rebin=2048)
rs.preprocessing_defaults()
```

## Sonification methods

| Method | Input | Mapping | Main controls | Extra |
|---|---|---|---|---|
| `profile` | 1-D, 2-D | interpolated profile waveform with an optional analytic instrument response | `sr`, `instrument` | base |
| `amplitude` | 1-D, 2-D | profile amplitude envelope on a harmonic carrier | `sr`, `freq`, `compression`, `harmonics`, `harmonic_decay` | base |
| `erb` | 2-D | time to time, ordered features to perceptual pitch, brightness and temporal salience to amplitude | frequency, contrast, timbre, envelope, event, and level settings | base |
| `griffinlim` | 2-D | mel-like magnitude interpretation with deterministic iterative phase reconstruction | `sr`, `n_iter`, `n_fft`, `frame_length`, `preemphasis`, `max_db`, `ref_db` | base |
| `hifigan` | 2-D | pinned checkpoint adapter and HiFi-GAN vocoder | registered model geometry | `hifigan` |
| `spatial_erb` | 3-D | one ERB synthesis per layer with constant-power stereo panning | ERB controls, `pan_positions`, `layer_gains` | base |

The ERB methods use overlapping normalized perceptual bands, continuous phase carriers, deterministic timbres, optional event accents, envelope smoothing, auditory-level compensation, RMS control, and true-peak limiting.

When provided, `pan_positions` and `layer_gains` use reusable sequences with one finite value per layer; pan positions range from `-1` to `1`, and gains are `0` or greater.

`timbre` selects the carrier waveform from `sine`, `retro_digital`, `warm_pad`, `soft_marimba`, `glass_bell`, and `instrument_palette`. `event_voice` selects `none` or `water_drop`. Advanced waveform and event controls live in the `voice_params` and `event_params` mappings, which both ERB methods share:

| Mapping | Key | Default | Accepted range |
|---|---|---|---|
| `voice_params` | `harmonic_limit_hz` | `3500.0` | above `0`, capped at `0.475 × sr` |
| `voice_params` | `detune_cents` | `10.0` | `0` to `50` |
| `voice_params` | `fm_index` | `1.0` | `0` to `1` |
| `voice_params` | `chorus_rate_hz` | `0.45` | `0` to `10` |
| `voice_params` | `chorus_depth_ms` | `8.0` | `0` to `20` |
| `event_params` | `salience_threshold` | `0.35` | `0` to `1` |
| `event_params` | `max_events_per_second` | `6.0` | `0` to `100` |
| `event_params` | `decay_ms` | `70.0` | `1` to `5000` |
| `event_params` | `level_db` | `-20.0` | `0` and below |

`harmonic_limit_hz` bounds the overtones of every timbre. `detune_cents`, `chorus_rate_hz`, and `chorus_depth_ms` shape `retro_digital`, `warm_pad`, and `instrument_palette`; `fm_index` shapes `retro_digital`. The `event_params` keys take effect once `event_voice="water_drop"` is selected.

```python
result = rs.sonify(
    matrix,
    data_duration=2.4,
    method="erb",
    method_params={
        "timbre": "glass_bell",
        "event_voice": "water_drop",
        "voice_params": {"harmonic_limit_hz": 6000.0},
        "event_params": {"max_events_per_second": 3.0, "level_db": -14.0},
    },
)
```

Griffin-Lim derives time and feature geometry from its FFT settings. HiFi-GAN accepts the shared normalized matrix and applies its checkpoint-specific 80-bin encoding internally.

## Optional postprocessors

`musicnet` applies one of six pretrained WaveNet music styles at 16 kHz. RadioSonify validates its dependencies and pinned model assets before primary synthesis, pads the final encoder window, and crops the decoded audio to the exact input span.

`rave` applies a trusted user-supplied TorchScript model and reads the model's sample-rate and channel metadata. RAVE TorchScript loading executes model code, so each RAVE export should come from a trusted source.

```python
styled = rs.sonify(
    profile,
    data_duration=2.0,
    method="amplitude",
    postprocess="musicnet",
    postprocess_params={"decoder_id": 2, "seed": 0},
)
```

## Timing and output

Every dimensionality, primary method, and postprocessor uses the same requested-duration formula:

```text
target_duration = data_duration × repeat ÷ speed
```

`speed=2` produces half the duration, and `speed=0.5` produces twice the duration. The registered repeat default is `5` for `amplitude` and `1` for the other primary methods. An explicit `repeat` controls every method.

`preserve_pitch=True` uses phase-vocoder time stretching. The standard polyphase path changes playback speed and pitch together. `output_sr` converts the final container rate while preserving physical duration and pitch. The final sample count is `round(sample_rate * target_duration)`.

Output conditioning removes DC, applies short edge fades, and constrains the waveform to the WAV range. Saving creates parent directories and writes a WAV file after path validation.

## Result and provenance

`sonify` returns a frozen `SonificationResult`. Its audio array and numeric array metadata use immutable buffers.

| Fields | Meaning |
|---|---|
| `audio`, `sample_rate`, `output_duration`, `output_path` | final waveform and container information |
| `data_type`, `data_duration`, `input_shape`, `source_name` | source identity and physical span |
| `source_time_axis`, `source_layer_axis` | resolved axes in the caller's original layout |
| `method`, `preprocess_params`, `method_params` | selected mapping and effective settings |
| `speed`, `repeat`, `preserve_pitch`, `target_duration` | timing controls |
| `method_sample_rate`, `method_native_samples`, `method_native_duration`, `method_time_scale` | primary synthesis timing |
| `postprocess`, `postprocess_params`, `postprocess_native_samples`, `postprocess_native_duration`, `postprocess_time_scale` | optional style-stage timing and settings |

Parameter mappings are recursively copied and frozen. HiFi-GAN also records its data-dependent `histogram_offset` in `method_params`.

## Low-level API and CLI adapters

The package exposes direct method functions for experiments that manage preprocessing and timing explicitly:

```python
rs.profile_to_wave(...)
rs.amplitude_modulate(...)
rs.erb_sonify(...)
rs.griffinlim_reconstruct(...)
rs.hifigan_vocode(...)
rs.spatial_sonify(...)
rs.musicnet_transform(...)
rs.rave_transform(...)
```

Method-specific CLI adapters are available as `profile`, `amplitude`, `erb`, `spatial-erb`, `griffinlim`, `hifigan`, `musicnet`, and `rave`. Griffin-Lim accepts the deprecated `--n-mels`, `--freq-rebin`, and `--time-rebin` aliases and routes them into shared preprocessing. New scripts can use `--preprocess` directly. [CHANGELOG.md](CHANGELOG.md) records public migrations and deprecations.

## Scientific use

Sonification is an interpretive representation of numerical structure. Baseline correction, clipping, normalization, resampling, synthesis, and neural style transfer each affect the audible result. Preserve the source data and record the RadioSonify version, input checksum, physical duration, axis declarations, effective parameters, model revision, and output sample rate for reproducible work.

The `profile`, `amplitude`, `erb`, and `spatial_erb` methods provide deterministic signal-processing mappings. Griffin-Lim uses deterministic phase initialization. MusicNet supports a recorded random seed. RAVE behavior follows the supplied model.

## Development and license

Development setup, validation commands, and contribution rules live in [CONTRIBUTING.md](CONTRIBUTING.md).

MSP-authored code uses the [MIT License](LICENSE). The included MusicNet inference subset and checkpoints use CC BY-NC 4.0 with a non-commercial-use condition. Distribution metadata uses `MIT AND CC-BY-NC-4.0`. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) provides component-level terms for each vendored component.
