# Model and data asset provenance

RadioSonify downloads optional resources from
[`TorchLight/radiosonify`](https://huggingface.co/TorchLight/radiosonify).
The client pins an immutable commit in `radiosonify.hub.REVISION`; a moving
branch is never used at runtime.

| Asset path | Origin and modifications | License |
|---|---|---|
| `data/Burst.npy`, `data/RawBurst.npy`, `data/ParkesBurst.npy`, `data/Profile.npy`, `data/Burst-wirfi.wav` | Example artifacts shipped in the original MSP repository under its top-level license. They are provided only as small API demonstrations, not as a scientific reference dataset. | MIT |
| `models/hifigan/config.json`, `models/hifigan/generator.pth` | HiFi-GAN Universal V1 architecture and base checkpoint from [`jik876/hifi-gan`](https://github.com/jik876/hifi-gan), then fine-tuned by the MSP authors for 500k steps on a collection of 500 symphonic recordings. The historical repository did not retain a machine-readable corpus manifest, so users requiring training-data-level provenance should not treat this checkpoint as fully auditable. | MIT for the upstream implementation and the MSP checkpoint, to the extent of the rights held by their respective authors |
| `models/musicnet/args.json`, `models/musicnet/{bestmodel,lastmodel}_{0..5}.pth` | Official pretrained MusicNet archive published by Facebook Research for [A Universal Music Translation Network](https://github.com/facebookresearch/music-translation). `args.json` is a safe JSON conversion of the original serialized argument object. | CC BY-NC 4.0; non-commercial use only |

On 2026-08-01, all twelve hosted MusicNet checkpoint files were verified as
follows: their byte size and ZIP CRC match Facebook Research's official
`pretrained_musicnet.zip`; their SHA-256 values match the files retained in the
MSP Git history and the pinned Hugging Face objects.

## Instrument responses

RadioSonify 0.2.0 no longer downloads the historical `piano.wav` and `vio.wav`
files because their external provenance could not be reconstructed. The
`piano` and `violin` responses are now generated deterministically from
analytic waveforms by `radiosonify.hub`, then cached locally as PCM16 WAV files.
No third-party sound recording is used.

## Citation

When publishing results, record the RadioSonify version, the pinned Hub
revision, the selected method/checkpoint, all effective parameters, the input
shape, and the input data's own provenance. The result metadata records most
runtime parameters but is not a substitute for preserving the original input
and its checksum.
