# Vendored model policy

The `hifigan/` and `musicnet/` packages contain checkpoint-compatible neural
network layers derived from their respective upstream projects. HiFi-GAN is
MIT; MusicNet is CC BY-NC 4.0 and therefore carries a non-commercial
restriction. They are not general RadioSonify application code.

- Keep only the inference surface required to load the published checkpoints.
- Preserve upstream copyright and license files.
- Avoid style-only rewrites that obscure comparison with upstream model
  definitions.
- Validate behavior through wrapper tests, import/compile checks, static
  inference-surface tests, and forward-shape contract tests with the relevant
  optional dependencies installed.

For those reasons this directory is intentionally excluded from the repository
Ruff pass. All package-owned wrappers outside this directory remain subject to
the full Ruff lint and formatting gates.
