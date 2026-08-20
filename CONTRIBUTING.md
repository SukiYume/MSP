# Contributing

## Development setup

MSP supports Python 3.9 through 3.13. Create and activate an isolated environment, then install the package and development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install all optional inference dependencies when working on neural backends:

```bash
python -m pip install -e ".[all,dev]"
```

## Module boundaries

- `validation.py` owns scalar, mapping, and array validation. `array_ops.py` owns generic array transforms, `audio_io.py` owns WAV paths and writes, and `runtime.py` owns optional dependency and temporary runtime-state helpers.
- `inputs.py` owns input types, source-axis declarations, canonical layout, and immutable source snapshots.
- `preprocessing.py` owns layer/time/feature resizing, scientific calibration, clipping, repetition, smoothing, and normalization. Primary methods consume its normalized output.
- `registry.py` owns declarative method and postprocessor metadata, defaults, validators, preflights, geometry callbacks, channel capabilities, and lazy runner resolution.
- `planning.py` resolves the complete immutable execution plan before array processing. `pipeline.py` executes that plan. `api.py` assembles public provenance and remains a thin facade over those two stages.
- `_perceptual_config.py` is the single source for shared ERB settings. `_perceptual.py` owns the filterbank, synthesis, and auditory conditioning engine; `_voices.py` owns sustained timbres and palette crossfades; `_events.py` owns local-peak selection and transient rendering.
- `erb.py` and `spatial.py` adapt normalized 2-D and 3-D arrays to the shared perceptual engine. Their scientific resizing stays in `preprocessing.py`.
- `timing.py` owns waveform duration fitting, sample-rate conversion, fades, and DC conditioning. `hifigan.py`, `musicnet.py`, and `rave.py` own their model adapters and runtime contracts. `models/` contains checkpoint-compatible vendored inference code.

## Validation

Run the same core gates used by CI:

```bash
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m mypy
python -m vulture
python -m pytest -q --cov=radiosonify --cov-report=term-missing --cov-fail-under=90
python -m build
python -m twine check dist/*
```

Changes to optional model wrappers or vendored inference code should also run:

```bash
python -m pytest -q tests/test_hifigan.py tests/test_musicnet.py tests/test_rave.py tests/test_vendored_models.py
```

## Pull request checklist

- Keep changes focused and preserve scientific input/timing contracts.
- Keep scientific-array resizing, baseline correction, clipping, and normalization in `preprocessing.py`; method implementations consume the shared `[0, 1]` contract.
- Resolve public policy in `planning.py` and keep `pipeline.py` focused on execution of an already validated plan.
- Add or update regression tests for behavior changes.
- Run the lint, formatting, and test gates above.
- Update `README.md`, `README_CN.md`, and `CHANGELOG.md` when public behavior
  changes.
- Keep downloaded checkpoints, generated audio, and observation data outside the repository.
- Keep `THIRD_PARTY_NOTICES.md`, `MODEL_ASSETS.md`, the adjacent vendored
  licenses, and the Hugging Face model card synchronized when assets change.

## Style and vendored code

Use explicit validation and actionable error messages. Code in
`src/radiosonify/models/` is a deliberately limited, checkpoint-compatible
inference surface; follow the policy in its `VENDORED.md` files and retain
upstream license notices when modifying it.

Before tagging a release, build from a clean commit, install the wheel into a
fresh virtual environment, run `radiosonify list-methods`, and verify that both
the MIT and CC BY-NC 4.0 license texts are present in the wheel. CI performs the
same release-artifact checks. Changes to the Hub compatibility layer must also
be tested against the declared minimum `huggingface_hub` version.
