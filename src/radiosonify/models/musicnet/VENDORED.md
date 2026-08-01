# Vendored MusicNet inference subset

These files derive from Facebook Research's Music Translation/WaveNet
implementation and remain covered by the complete CC BY-NC 4.0 text in the
adjacent `LICENSE`. The attribution and non-commercial restriction apply to
this directory and the corresponding pretrained checkpoints.

RadioSonify vendors only the code needed to construct the released encoder and
decoder checkpoints and run inference. Training-only discriminators, losses,
weight-export helpers, profiling utilities, and unreachable constructor modes
are intentionally removed. Local changes also maintain streaming
autoregressive state between decode splits.

When updating this directory:

1. preserve state-dict module names and tensor shapes used by the released
   checkpoints;
2. retain the upstream copyright and license headers;
3. keep training/export utilities outside the runtime package; and
4. run the MusicNet, API, compile, and static vendored-contract tests.

Upstream: <https://github.com/facebookresearch/music-translation>

Upstream citation: Noam Mor, Lior Wolf, Adam Polyak, and Yaniv Taigman,
"A Universal Music Translation Network," ICLR 2019.
