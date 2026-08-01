# Contributing

## Development setup

MSP supports Python 3.9 through 3.13. Create and activate an isolated
environment, then install the package and development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install all optional inference dependencies only when working on neural
backends:

```bash
python -m pip install -e ".[all,dev]"
```

## Validation

Run the same core gates used by CI:

```bash
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m pytest -q
python -m build
python -m twine check dist/*
```

Changes to optional model wrappers or vendored inference code should also run:

```bash
python -m pytest -q tests/test_hifigan.py tests/test_musicnet.py tests/test_vendored_models.py
```

## Pull request checklist

- Keep changes focused and preserve scientific input/timing contracts.
- Add or update regression tests for behavior changes.
- Run the lint, formatting, and test gates above.
- Update `README.md`, `README_CN.md`, and `CHANGELOG.md` when public behavior
  changes.
- Do not commit downloaded checkpoints, generated audio, or observation data.
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
