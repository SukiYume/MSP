# Third-party notices

RadioSonify/MSP is a mixed-license distribution. Code written for MSP is
licensed under the repository's `LICENSE` (MIT). The optional MusicNet
inference subset is not MIT and makes the distribution as a whole subject to
both licenses declared in `pyproject.toml`.

## HiFi-GAN

- Upstream: <https://github.com/jik876/hifi-gan>
- Authors: Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae
- License: MIT
- Local scope: `src/radiosonify/models/hifigan/`
- Local changes: the model definition is reduced to the checkpoint-compatible
  inference surface and adapted to explicit channel bookkeeping.

The complete upstream MIT text is retained at
`src/radiosonify/models/hifigan/LICENSE`.

## A Universal Music Translation Network

- Upstream: <https://github.com/facebookresearch/music-translation>
- Authors named by the upstream project: Noam Mor, Lior Wolf, Adam Polyak, and
  Yaniv Taigman
- License: Creative Commons Attribution-NonCommercial 4.0 International
  (`CC-BY-NC-4.0`)
- Local scope: `src/radiosonify/models/musicnet/` and the optional MusicNet
  checkpoints documented in `MODEL_ASSETS.md`
- Local changes: training/export/profiling code was removed; the retained
  encoder and WaveNet decoder were made checkpoint-compatible with current
  PyTorch, and split decoding preserves autoregressive state.

The complete CC BY-NC 4.0 text is retained at
`src/radiosonify/models/musicnet/LICENSE`. MusicNet code and checkpoints may be
used and redistributed only under those terms, including the non-commercial
restriction and attribution requirements.

## Remote resources

Remote example data and neural weights are not implicitly relicensed by the
software's MIT license. Their exact origin, scope, and license are listed in
`MODEL_ASSETS.md` and in the model card at
<https://huggingface.co/TorchLight/radiosonify>.
