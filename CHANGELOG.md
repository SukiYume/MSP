# Changelog

All notable changes to this project are documented in this file.

## [0.1.2] - 2026-03-05

### Fixed
- Validate `rebin_spectrogram()` target bins to prevent upsampling reshape crashes.
- Correct `amplitude_modulate()` carrier generation so `freq` maps to physical Hz.
- Add explicit CUDA availability check in `musicnet()` with clear error message.
- Replace `copy.deepcopy()` with `ndarray.copy()` in Griffin-Lim path.
- Replace deprecated `F.tanh` with `torch.tanh` in MusicNet model code.

### Security
- Use `torch.load(..., weights_only=True)` where supported, with backward-compatible fallback.

### Improved
- Read instrument WAV files via `soundfile` for robust format handling.
- Support cache directory override with `RADIOSONIFY_CACHE_DIR`.
- Stronger CLI input path validation.
- Add `py.typed` marker for type-checking consumers.
- Add missing tests for `musicnet.py`.
- Add GitHub Actions CI workflow.
