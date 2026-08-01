# Changelog

All notable changes to this project are documented in this file.

## [0.2.0] - 2026-08-01

### Added

- Add the duration-aware `sonify()` API, immutable input/result records, method
  registry, postprocessor registry, and reproducibility metadata.
- Add amplitude compression, exact duration fitting, and optional
  pitch-preserving time stretching.
- Add `output_sr` to normalize final WAV container rates without changing
  physical pitch or inventing high-frequency content.
- Add lazy public exports and a runnable unified-API example.
- Add Python 3.9–3.13 CI plus optional neural-backend contract tests.
- Add release-artifact, minimum-dependency, clean-wheel-install, CLI, license,
  and model-asset provenance checks.

### Changed

- Treat MusicNet as a native-16-kHz optional audio postprocessor and apply
  playback speed only after autoregressive generation.
- Reduce the Griffin–Lim default to 64 iterations, make `preemphasis=0` the
  scientific default, and standardize on `freq_rebin`.
- Keep heavy SciPy, librosa, Hugging Face Hub, and neural-model imports lazy.
- Trim vendored model code to the checkpoint-compatible inference surface.
- Replace the historical piano/violin recordings of unknown provenance with
  deterministic, locally generated instrument responses.
- Declare the distribution's actual composite license as
  `MIT AND CC-BY-NC-4.0`; MSP code remains MIT while the MusicNet inference
  subset and checkpoints remain non-commercial CC BY-NC 4.0.

### Compatibility

- Keep the `radiosonify` 0.1.x console command and its method subcommands.
- Accept the old low-level `musicnet(batch_size=...)` argument with a
  deprecation warning. It is ignored because the entry point handles one
  recording; the new unified API does not expose it.
- Keep the Griffin–Lim CLI `--n-mels` option as a deprecated alias for
  `--freq-rebin`.

### Fixed

- Prevent duration fitting from mutating caller-owned arrays.
- Preserve the full input extent during rebinning and handle very short output
  arrays without silencing them.
- Correct Hugging Face online/offline retry exception handling.
- Validate all controls, method parameters, and output paths before expensive
  synthesis starts.
- Support the declared `huggingface_hub>=0.20` lower bound by importing Hub
  exceptions from the version-compatible public namespace.
- Validate output paths at the start of every public low-level method, before
  input preparation, resource downloads, or model inference.

## [0.1.2] - 2026-03-05

### Fixed

- Validate `rebin_spectrogram()` target bins to prevent upsampling reshape
  crashes.
- Correct `amplitude_modulate()` carrier generation so `freq` maps to physical
  Hz.
- Add an explicit CUDA availability check in `musicnet()` with a clear error
  message.
- Replace `copy.deepcopy()` with `ndarray.copy()` in the Griffin–Lim path.
- Replace deprecated `F.tanh` with `torch.tanh` in MusicNet model code.

### Security

- Load model checkpoints with `torch.load(..., weights_only=True)`.

### Improved

- Read instrument WAV files with `soundfile` for robust format handling.
- Support cache-directory overrides through `RADIOSONIFY_CACHE_DIR`.
- Add the `py.typed` marker for type-checking consumers.
