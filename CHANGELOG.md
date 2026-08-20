# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [0.3.0] - 2026-08-20

### Added

- Add immutable execution planning before scientific-array processing. Method validators, geometry checks, output checks, optional dependency checks, model-asset resolution, postprocessor channel checks, and RAVE TorchScript contract inspection now finish before preprocessing begins.
- Add `layer_rebin` for ordered area-average reduction of three-dimensional layer stacks. Spatial pan and gain controls validate against the planned layer count.
- Add a real TorchScript RAVE contract test, a 90% owned-code coverage gate, a McCabe complexity limit of 10, Vulture dead-code scanning, and repository-wide LF normalization.

- Register the grouped `voice_params` and `event_params` defaults on the ERB
  method specs, and expand them in `list-settings`. The two mappings default to
  `None`, so the nine advanced waveform and event settings were previously
  reachable only by reading the source.

- Add `scale_statistic` (`'std'` / `'mad'`) to the shared preprocessing stage.
  It divides each channel by its own noise scale after baseline correction,
  which is what makes a burst audible over bandpass and RFI: measured on
  FRB180301, the ratio between the noisiest and quietest channel drops from
  17.3 to 1.8.
- Add `nan_policy='propagate'` so masked channels are handled with nan-aware
  statistics and become silence after normalization. Infinite values are
  rejected under every policy.
- Add `normalization_scope`, defaulting to `per_layer` for 3-D input. Global
  min-max let the strongest layer decide everyone else's loudness; on a
  simulated I/Q/U/V cube the weak layers came out 300x quieter.
- Add public `time_rebin`, `feature_rebin` and `time_smoothing` preprocessing
  parameters, so every shape and filtering decision is configurable from the
  one stage that owns data.
- Add `time_axis` and `layer_axis` to `SonificationInput`. Axis semantics are an
  input contract, not a method parameter: the shared baseline is computed per
  channel along the time axis, so a method-level axis switch silently produced a
  baseline along the wrong axis.
- Add a `sonify` CLI command exposing the unified pipeline, plus `list-settings`
  and repeatable preprocessing, method, and postprocessor settings.
- Add `time_rebin='auto'` support to Griffin-Lim through registered frame
  geometry. It previously synthesized at its input's native length and was then
  resampled: at `speed=0.5` that was a 6.3x stretch, shifting pitch and
  collapsing bandwidth.
- Add an early channel-compatibility check between methods and postprocessors,
  so a 3-D stereo result is rejected before synthesis rather than after it.
- Add one public, method-independent preprocessing stage for 1-D, 2-D, and
  3-D inputs: configurable subtract/divide baseline correction with mean/median,
  percentile clipping, and mandatory min-max normalization to `[0, 1]`.
- Add domain-neutral 2-D `erb` sonification with configurable frequency
  direction, amplitude/power interpretation, and phase-continuous perceptual
  oscillators. Public `mel_frequencies` and `erb_frequencies` helpers expose
  both supported center-frequency scales.
- Add deterministic `warm_pad`, `soft_marimba`, and `glass_bell` ERB voices,
  plus an `instrument_palette` that crossfades them continuously from low to
  high pitch while preserving the brightness envelope.
- Add an opt-in `water_drop` event layer driven by an independent temporal-
  salience map, with bounded event density, threshold, decay, and level controls.
- Add 3-D `spatial_erb` sonification for generic layer stacks, including I/Q/U/V
  data, consuming the shared preprocessing output and providing configurable
  stereo pan positions and gains.
- Add optional RAVE TorchScript postprocessing using a trusted user-supplied
  model, including model sample-rate discovery and stereo-preserving use of
  mono models.
- Add `matrix`, `image`, `layered_matrix`, `cube`, and `iquv` input aliases while
  retaining `dynamic_spectrum` compatibility.
- Add full multi-channel duration fitting, conditioning, provenance, and WAV
  output support.
- Add `input_shape`, `source_time_axis`, and `source_layer_axis` provenance to
  `SonificationInput` and `SonificationResult`.
- Add mypy to the development dependencies and CI quality gates.

### Changed

- Record grouped method parameters at the same resolution as top-level ones.
  A result listed all nineteen registered ERB settings while storing only the
  caller's `voice_params` subset, so one record carried two conventions. A
  misspelled key inside a group is now also rejected before synthesis starts.

- Merge the three copies of settings validation into one `validation._merge_settings`.
  Method parameters, preprocessing parameters, postprocessor parameters and the
  grouped `voice_params`/`event_params` mappings now report an unrecognized key
  the same way, including the list of accepted names.
- Print every registered method and postprocessor default in `list-settings`, and name the required install extra in `list-methods`.
- Split the former `core.py` catch-all into `validation.py`, `array_ops.py`, `audio_io.py`, and `runtime.py`. Split public orchestration into plan resolution in `planning.py`, execution in `pipeline.py`, and result assembly in `api.py`.
- Route command-line execution through `radiosonify sonify`, with repeatable `--preprocess`, `--method-param`, and `--postprocess-param` settings.
- Resolve `RADIOSONIFY_CACHE_DIR` on each call instead of at import, so a
  process that configures the cache after importing the package still
  redirects its downloads and generated instrument responses.
- Document that `baseline_axis` also selects the per-channel direction used by
  `scale_statistic`, and that percentile clipping is measured across the whole
  array while `normalization_scope` governs only the following min-max step.
- Keep all scientific resizing in `preprocess()` and record its effective settings in every result.
- Validate MusicNet's 800-sample encoder minimum from the planned primary-audio length before preprocessing or model loading.
- Include every document linked from the README in the source distribution and verify that inventory in CI.

- Rebuild the `water_drop` event voice around a quiet band-limited impact and a
  delayed, damped bubble resonance at the mapped pitch. Coordinate-derived
  micro-variation keeps repeated events organic and fully deterministic.
- Refactor `erb` and `spatial_erb` around one continuous ambient-detail mapping.
  Normalized triangular feature bands contribute low-level absolute-brightness
  ambience plus positive deviation above each band's temporal median and MAD
  scale. `n_bands=None` derives approximately one simultaneous voice per ERB;
  the default 100 to 2000 Hz range resolves to 18. Logarithmic HTK-mel centers,
  deterministic quadratic phases, fixed mix normalization, 8/80 ms envelope
  smoothing, bounded equal-loudness compensation, a -20 dBFS RMS ceiling, and a
  four-times-oversampled -1 dBFS true-peak ceiling define the standard path.
  The default carrier remains sine, optional timbres share the same envelopes,
  and optional events add accents from the shared salience map. Advanced
  waveform and event controls live in `voice_params` and `event_params`.
- Validate normalized ERB arrays once at the public adapters, reuse one temporal
  salience map for continuous and optional event layers, and keep timbre and
  event choices in one shared configuration source.
- Keep isolated builds on Hatchling 1.27 so Python 3.9 remains supported and
  release artifacts use Core Metadata 2.4 accepted by the current Twine gate.
- Generalize `repeat` to every dimensionality and method by applying it in
  preprocessing rather than inside the profile methods.
- Reorder the preprocessing pipeline to rebin first. Measured on FRB180301
  (28346x4096 to 2048x512), rebinning before baseline/clipping leaves the burst
  occupying 4.5x more of the output's dynamic range at identical SNR, because
  clip bounds computed at input resolution are set by single-sample noise.
- Move every remaining data-domain knob out of the methods:
  `time_downsample`, `time_rebin`, `freq_rebin`/`n_mels`, `time_smoothing`,
  `time_axis` and `layer_axis` are method-independent settings, and method
  parameter validation rejects these retired names.
- Keep percentile clipping available as an explicit preprocessing option while
  leaving it disabled by default, so the standard pipeline preserves the full
  input value range.
- Allow baseline correction to be disabled for calibrated matrices and display
  images whose absolute intensity should control loudness.
- Report HiFi-GAN's data-dependent histogram offset through a `provenance`
  out-parameter so it is recorded in the result instead of being invisible.
- Keep the checkpoint's fixed encoding inside the method that owns it.
  HiFi-GAN leaves shared `input_feature_bins` free and receives the normalized
  scientific matrix, then internally performs the historical resize-to-80,
  range restoration, histogram alignment, `* 12 - 10.5` mapping, and
  `[-11, 1.6]` clipping required by its checkpoint.
- Make `DataType.MATRIX` its own value (`'matrix'`). It previously shared
  `'dynamic_spectrum'` with a second member, which Python folded into an alias,
  so registry entries that looked like different input constraints were
  identical.
- Use one scientific-array resize implementation in preprocessing. Extract
  ERB and spatial ERB defaults into one immutable configuration and their
  shared filterbank, timbre, and conditioning code into an internal engine;
  the public modules are now independent 2-D and 3-D adapters.
- Remove the unused ERB feature-axis resampler and deduplicate the shared ERB
  CLI options, registry defaults, and adapter argument forwarding.
- Make ERB synthesis the dependency-free default for 2-D arrays and spatial
  ERB synthesis the default for 3-D arrays; neither assumes DM or astronomy.
- Use full-coverage equal-width area rebinning whenever ERB time or feature
  axes must be downsampled, including non-divisible dimensions.
- Standardize profile amplitude sonification on a linear envelope, five
  repeats, and a four-partial fixed carrier with `1/h` harmonic decay. Set
  `harmonics=1` or an explicit `repeat` to select the former behavior.
- Omit requested harmonics at or above Nyquist instead of aliasing them.
- Validate the RAVE optional dependency and requested CPU/CUDA/MPS device before
  primary sonification begins.
- Move scientific-array normalization out of every sonification method and
  record the effective `preprocess_params` in `SonificationResult`. Low-level
  methods now consume already-normalized arrays; the unified API and CLI apply
  shared preprocessing before dispatch.
- Remove method-local ERB `normalization` settings and neural `clean` /
  `exposure_cut` switches. Equivalent conditioning belongs in the shared
  preprocessing stage.
- Make unified HiFi-GAN input length duration-aware by default:
  `time_rebin="auto"` resolves to `round(target_duration * 22050 / 256)`, capped
  at the source time dimension, while the result records the resolved integer.
- Preserve raw HiFi-GAN generator gain during final DC/fade conditioning rather
  than normalizing quiet neural background to a `0.9` peak.
- Give same-named submodules standard Python import behavior and keep the package
  root focused on the unified API, discovery, preprocessing, timing, and I/O.
- Rewrite the English and Chinese user guides around installation, the unified
  API, the full preprocessing contract, methods, timing, and provenance.
- Use shared `preprocess()` settings as the maintained conditioning path.

### Removed

- Remove the deprecated `del_burst()` and `rebin_spectrogram()` helpers, package-root method aliases, method-name aliases, method-specific CLI commands, Griffin-Lim compatibility options, HiFi-GAN method-local resizing and smoothing controls, and the ignored MusicNet `batch_size` argument.
- Remove `MethodSpec.label`, `MethodSpec.model_feature_bins`, and the single-valued `normalization` preprocessing setting. The checkpoint bin count remains owned by `radiosonify.hifigan`.
- Remove duplicate choice validators, duplicate FFT test helpers, unused HiFi-GAN configuration attributes, and obsolete resampling implementations.

### Fixed

- Parse string Torch device identifiers explicitly so `cuda:N` seeds the requested device index instead of reading `str.index` as metadata.
- Read the standard nn~ `sampling_rate` and four-value `forward_params`
  metadata from real exported RAVE models. Input and output channel counts are
  now independent, so official mono-input/stereo-output models no longer fail
  against the synthetic `sr`/`n_channels` contract previously used by tests.
- Record a RAVE inference seed and restore the caller's Torch RNG state after
  conversion, making stochastic exported models reproducible by default.

- Reject one-shot iterators and byte-oriented objects supplied as `spatial_erb`
  pan or layer-gain controls before synthesis, keeping rendered values and result
  provenance aligned.

- Freeze every `Sequence` parameter by value in the result record. The freezing
  logic matched `list` and `tuple` only, so a custom sequence such as
  `UserList` passed as `pan_positions` was stored by reference: editing the
  original afterwards changed `result.method_params`, contradicting the frozen
  result contract. Text, bytes, arrays, sets and mappings keep their existing
  treatment.

- Build immutable provenance snapshots straight from the strided view.
  `tobytes(order="C")` already serializes any layout in C order, so the extra
  contiguity pass only added a second full-size copy for exactly the
  transposed 3-D views the standard layout produces.

- Account for Griffin-Lim's shared ISTFT boundary frame when deriving automatic
  time geometry, including repeated inputs, so native synthesis reaches the
  requested duration before final fitting. A single-frame, single-pass request
  remains valid because boundary overlap applies only between repeated copies.
- Keep HiFi-GAN checkpoint loading on `weights_only=True` for every failure path;
  checkpoint decoding errors no longer trigger legacy pickle loading.
- Preflight MusicNet dependencies and pinned assets before primary synthesis;
  pad the final 800-sample encoder window and crop decoded audio to the exact
  validated input length.
- Back public input, result-audio, and numeric provenance arrays with immutable
  buffers so callers cannot re-enable NumPy writes.
- Keep extreme positive resampling ratios valid when bounded rational
  approximation would otherwise produce a zero numerator.
- Serialize same-process instrument-cache replacement on Windows and use a
  unique temporary filename for each writer.
- Convert expected library, dependency, resource, and validation failures into
  concise Click errors across the command-line interface.
- Resolve Griffin-Lim's default and maximum feature bins from `n_fft`, including
  early validation of explicit `feature_rebin` values.
- Renormalize NaN-aware downsampling and interpolation from valid overlap, so a
  masked sample stays local during axis resizing.
- Join repeated data before temporal smoothing, allowing the filter to cover
  internal repeat boundaries.
- Limit polyphase sample-rate conversion overshoot before RAVE inference and
  validate RAVE model metadata and decoded channel geometry strictly.
- Validate existing output-path parent components before synthesis and before
  optional model loading.
- Convert sample rates through one exact-count, peak-safe implementation across
  profile responses, MusicNet, RAVE, and `output_sr`.
- Convert extreme duration, repeat, frequency, and harmonic combinations into
  documented `ValueError` failures rather than leaking numeric overflow errors.
- Validate primary and postprocessed audio as finite, non-empty one- or two-dimensional arrays before duration calculations, and enforce each primary method's registered channel count.

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
