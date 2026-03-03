# AstroSonify PyPI Release Design

**Date**: 2026-03-03
**Package Name**: `astrosonify`
**Initial Version**: 0.1.0
**Python**: >= 3.9
**License**: MIT

## Overview

将 MSP (Methods for Sonifying Pulse) 从科研脚本集合重构为可通过 `pip install astrosonify` 安装的 Python 包，同时提供 Python API 和 CLI 入口。

## Project Structure

```
MSP/
├── pyproject.toml
├── LICENSE
├── README.md
├── README_CN.md
├── src/
│   └── astrosonify/
│       ├── __init__.py             # 版本号 + 公共 API 导出
│       ├── core.py                 # 公共工具：rebin_spectrogram, normalize, del_burst, read_wave
│       ├── astronify_method.py     # 方法 0：astronify 声化
│       ├── profile.py              # 方法 1：脉冲轮廓转波形
│       ├── amplitude.py            # 方法 2：振幅调制正弦波
│       ├── griffinlim.py           # 方法 3：Griffin-Lim 声码器
│       ├── hifigan.py              # 方法 4：HiFi-GAN 神经声码器
│       ├── musicnet.py             # 方法 5：WaveNet 音乐风格迁移
│       ├── hub.py                  # Hugging Face 模型/数据下载管理器
│       ├── cli.py                  # CLI 入口 (click)
│       └── models/                 # 模型定义代码（不含权重）
│           ├── __init__.py
│           ├── hifigan/
│           │   ├── __init__.py
│           │   ├── generator.py    # Generator, weight_norm 等
│           │   └── env.py          # AttrDict
│           └── musicnet/
│               ├── __init__.py
│               ├── wavenet.py
│               ├── wavenet_models.py
│               ├── wavenet_generator.py
│               └── utils.py
├── tests/
│   ├── test_core.py
│   ├── test_profile.py
│   ├── test_amplitude.py
│   ├── test_griffinlim.py
│   ├── test_hub.py
│   └── test_cli.py
├── Data/                           # 本地开发用（不打包）
├── Figure/
└── Instruments/                    # 乐器采样（打包进 sdist）
```

## Dependencies

### Core Dependencies
- numpy
- scipy
- librosa
- soundfile
- huggingface_hub
- click

### Optional Dependencies
```toml
[project.optional-dependencies]
astronify = ["astropy", "astronify"]
hifigan = ["torch", "scikit-image"]
musicnet = ["torch", "tqdm"]
all = ["astropy", "astronify", "torch", "scikit-image", "tqdm"]
```

## Python API

### Unified Design Principles
1. 输入统一为 numpy 数组
2. 返回 `(audio_array, sample_rate)` 元组
3. 可选 `output` 参数写文件
4. 内部自动适配维度（rebin/resize），用户可通过参数手动控制
5. 缺少可选依赖时给出清晰 ImportError

### Public API

```python
import astrosonify as asf

# 方法 0：astronify
audio, sr = asf.astronify_sonify(data, note_spacing=0.01, output=None)

# 方法 1：脉冲轮廓转波形
audio, sr = asf.profile_to_wave(data, sr=48000, duration=10,
                                 time_downsample=10, instrument="violin", output=None)

# 方法 2：振幅调制
audio, sr = asf.amplitude_modulate(data, sr=48000, duration=2, freq=1000, output=None)

# 方法 3：Griffin-Lim
audio, sr = asf.griffinlim(spectrogram, sr=48000, n_iter=200, n_mels=512,
                            time_rebin=None, freq_rebin=None, output=None)

# 方法 4：HiFi-GAN
audio, sr = asf.hifigan(spectrogram, time_rebin=None, output=None)

# 方法 5：MusicNet
audio, sr = asf.musicnet(input_wav_or_array, decoder_id=2,
                          checkpoint_type="bestmodel", sr=48000, output=None)

# 便捷函数
data = asf.load_example("burst")           # 从 HF 下载示例数据
asf.save_audio(audio, sr, "output.wav")    # 保存音频
```

### Input Dimension Handling

- **Profile 方法（0, 1, 2）**：接受 1D 或 2D。2D 自动沿频率轴平均。`time_downsample` 控制时间降采样。
- **Spectrogram 方法（3, 4）**：接受 2D (time × freq)。内部自动 rebin 到目标维度。用户可通过 `time_rebin` / `freq_rebin` 手动控制。
- **MusicNet（5）**：接受 WAV 文件路径或 1D 音频数组。

`core.py` 提供 `rebin_spectrogram(data, time_bins, freq_bins)` 统一工具。

## CLI

```bash
astrosonify <method> --input data.npy --output audio.wav [options]

# 子命令
astrosonify griffinlim --input burst.npy --output burst.wav --sr 48000
astrosonify hifigan --input raw_burst.npy --output burst_hifigan.wav
astrosonify profile --input burst.npy --output profile.wav --instrument violin
astrosonify amplitude --input profile.npy --output amp.wav --freq 1000
astrosonify download-examples --dest ./data/
astrosonify list-methods
```

使用 `click` 库，方法名作为子命令。

## Hugging Face Hub Integration

**HF 仓库**: `SukiYume/astrosonify`

```
astrosonify/
├── data/
│   ├── Burst.npy
│   ├── RawBurst.npy
│   ├── ParkesBurst.npy
│   └── Profile.npy
├── models/
│   ├── hifigan/
│   │   ├── config.json
│   │   └── generator.pth
│   └── musicnet/
│       ├── args.pth
│       ├── bestmodel_0.pth ~ bestmodel_5.pth
│       └── lastmodel_0.pth ~ lastmodel_5.pth
└── instruments/
    ├── piano.wav
    └── vio.wav
```

- 使用 `huggingface_hub.hf_hub_download`
- 缓存到 `~/.cache/astrosonify/`
- lazy download（首次使用时下载）
- 支持离线模式

## Testing

- 基础方法用合成 numpy 数组测试
- HiFi-GAN / MusicNet 用 `@pytest.mark.skipif` 标记
- hub 下载用 mock
- CLI 用 `click.testing.CliRunner`

## Third-Party Code

- `models/hifigan/`: 改编自 [jik876/hifi-gan](https://github.com/jik876/hifi-gan) (MIT)
- `models/musicnet/`: 改编自 [facebookresearch/music-translation](https://github.com/facebookresearch/music-translation) (需确认许可)

需在包内保留原始许可声明。
