# Third-party notices and asset provenance

RadioSonify/MSP is a mixed-license distribution. Code written for MSP is
licensed under the repository's `LICENSE` (MIT). The optional MusicNet
inference subset is not MIT and makes the distribution as a whole subject to
both licenses declared in `pyproject.toml`.

## Hosted assets

RadioSonify downloads optional resources from the pinned immutable commit stored in `radiosonify.hub.REVISION` at [`TorchLight/radiosonify`](https://huggingface.co/TorchLight/radiosonify). Runtime downloads never follow a moving branch.

| Asset path | Origin and modifications | License |
|---|---|---|
| `data/Burst.npy`, `data/RawBurst.npy`, `data/ParkesBurst.npy`, `data/Profile.npy`, `data/Burst-wirfi.wav` | Example artifacts retained from the original MSP repository as small API demonstrations. | MIT |
| `models/hifigan/config.json`, `models/hifigan/generator.pth` | HiFi-GAN Universal V1 architecture and base checkpoint from [`jik876/hifi-gan`](https://github.com/jik876/hifi-gan), followed by 500k MSP fine-tuning steps on a collection of 500 symphonic recordings. The historical repository has no machine-readable corpus manifest, which limits training-data-level auditability. | MIT for the upstream implementation and the MSP checkpoint, to the extent of the rights held by their respective authors |
| `models/musicnet/args.json`, `models/musicnet/{bestmodel,lastmodel}_{0..5}.pth` | Official pretrained MusicNet archive from Facebook Research's [A Universal Music Translation Network](https://github.com/facebookresearch/music-translation). `args.json` is a safe JSON conversion of the original serialized argument object. | CC BY-NC 4.0; non-commercial use only |

Remote example data and neural weights retain the terms listed here and in the [Hugging Face model card](https://huggingface.co/TorchLight/radiosonify).

## HiFi-GAN

- Upstream: <https://github.com/jik876/hifi-gan>
- Authors: Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae
- License: MIT
- Local scope: `src/radiosonify/models/hifigan/`
- Local changes: the model definition is reduced to the checkpoint-compatible
  inference surface and adapted to explicit channel bookkeeping.

The complete upstream MIT text is retained at
`src/radiosonify/models/hifigan/LICENSE`.

The fixed checkpoint adapter resizes the preprocessed feature axis to 80 bins, restores the resized matrix to `[0, 1]`, estimates its histogram mode `m`, and applies the historical mapping `12 * (x + 0.6 - m) - 10.5`, clipped to `[-11, 1.6]`. The equivalent unclipped expression is `12 * (x - m) - 3.3`. These empirical constants define the checkpoint input domain. `SonificationResult.method_params` records the data-dependent histogram offset.

## A Universal Music Translation Network

- Upstream: <https://github.com/facebookresearch/music-translation>
- Authors named by the upstream project: Noam Mor, Lior Wolf, Adam Polyak, and
  Yaniv Taigman
- License: Creative Commons Attribution-NonCommercial 4.0 International
  (`CC-BY-NC-4.0`)
- Local scope: `src/radiosonify/models/musicnet/` and the optional MusicNet checkpoints listed above
- Local changes: training/export/profiling code was removed; the retained
  encoder and WaveNet decoder were made checkpoint-compatible with current
  PyTorch, and split decoding preserves autoregressive state.

The complete CC BY-NC 4.0 text is retained at
`src/radiosonify/models/musicnet/LICENSE`. MusicNet code and checkpoints may be
used and redistributed only under those terms, including the non-commercial
restriction and attribution requirements.

On 2026-08-01, all twelve hosted MusicNet checkpoint files were verified against Facebook Research's official `pretrained_musicnet.zip`: byte sizes and ZIP CRC values match the archive, while SHA-256 values match the files retained in MSP Git history and the pinned Hugging Face objects.

## RAVE models

RAVE model files are supplied by the user. The `rave` postprocessor accepts a trusted TorchScript export and records its runtime contract; provenance and license remain attached to that selected model. `torch.jit.load` executes model code, so only trusted exports should be loaded.

## Generated instrument responses

RadioSonify 0.2.0 generates the `piano` and `violin` responses deterministically from analytic waveforms in `radiosonify.hub` and caches them as PCM16 WAV files. This replaces the historical `piano.wav` and `vio.wav` recordings whose external provenance could not be reconstructed.

## Scientific provenance

Published results should record the RadioSonify version, source checksum, pinned Hub revision, selected method and checkpoint, resolved parameters, input shape and axes, physical duration, input-data provenance, and output checksum. Runtime result metadata captures most execution parameters, while the original input and its checksum remain part of the research record.
