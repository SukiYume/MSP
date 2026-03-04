<div align="center">

<div align="center"><img style="border-radius:50%;border: royalblue dashed 1px;padding: 5px" src="Figure/Burst.png" alt="RMS" width="140px" /></div>

# AstroSonify

_多种方法声化射电脉冲_

</div>

<p align="center">
  <a href="https://pypi.org/project/astrosonify/">
    <img src="https://img.shields.io/pypi/v/astrosonify?color=royalblue" alt="PyPI">
  </a>
  <a href="https://github.com/SukiYume/MSP">
    <img src="https://img.shields.io/badge/MethodSonifyPulse-MSP-royalblue" alt="MSP">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
</p>

<p align="center">
  <a href="./README.md" target="_blank">English README</a>
</p>

## 简介

射电望远镜可以将电磁信号数字化采样并记录下来，但接收频率通常不在人耳可听范围内。原始数据通常经过傅立叶变换转换到时间-频率域，并丢弃相位信息以节省存储空间，因此无法恢复原始波形。

**AstroSonify** 提供 6 种方法将这类无相位的时频数据转换为可听声音，从简单的轮廓映射到神经声码器重建。

## 安装

```bash
# 核心包（方法 1-3）
pip install astrosonify

# 附带 HiFi-GAN 神经声码器（方法 4）
pip install astrosonify[hifigan]

# 附带 MusicNet 风格迁移（方法 5）
pip install astrosonify[musicnet]

# 附带 astronify（方法 0）
pip install astrosonify[astronify]

# 安装全部
pip install astrosonify[all]
```

## 快速开始

### Python API

```python
import astrosonify as asf

# 从 Hugging Face Hub 加载示例数据
data = asf.load_example("burst")        # 二维时频谱 (时间 x 频率)
profile = asf.load_example("profile")   # 一维脉冲轮廓

# 方法 1：脉冲轮廓转波形（小提琴音色卷积）
audio, sr = asf.profile_to_wave(data, sr=48000, duration=10, instrument="violin")

# 方法 2：振幅调制正弦波
audio, sr = asf.amplitude_modulate(profile, sr=48000, duration=2, freq=1000)

# 方法 3：Griffin-Lim 声码器
audio, sr = asf.griffinlim(data, sr=48000, n_iter=200)

# 方法 4：HiFi-GAN 神经声码器（需要 torch）
audio, sr = asf.hifigan(data)

# 方法 5：WaveNet 音乐风格迁移（需要 torch + CUDA）
audio, sr = asf.musicnet("input.wav", decoder_id=2)

# 保存输出
asf.save_audio(audio, sr, "output.wav")
```

### 命令行

```bash
# 列出可用方法
astrosonify list-methods

# 使用 Griffin-Lim 声化
astrosonify griffinlim --input burst.npy --output burst.wav --sr 48000

# 使用轮廓方法声化
astrosonify profile --input burst.npy --output profile.wav --instrument violin

# 使用振幅调制声化
astrosonify amplitude --input profile.npy --output amp.wav --freq 1000

# 下载示例数据
astrosonify download-examples --dest ./data/
```

## 方法列表

| # | 方法 | 函数 | 额外依赖 |
|---|------|------|---------|
| 0 | Astronify（音高映射）| `astronify_sonify()` | astropy, astronify |
| 1 | 脉冲轮廓转波形 | `profile_to_wave()` | 无 |
| 2 | 振幅调制 | `amplitude_modulate()` | 无 |
| 3 | Griffin-Lim 声码器 | `griffinlim()` | 无 |
| 4 | HiFi-GAN 神经声码器 | `hifigan()` | torch, scikit-image |
| 5 | WaveNet 风格迁移 | `musicnet()` | torch, tqdm, CUDA |

### 输入处理

- **轮廓类方法（0, 1, 2）**：接受一维轮廓或二维时频谱（自动沿频率轴平均）
- **时频谱方法（3, 4）**：接受二维时频谱（自动重采样到目标维度）
- **MusicNet（5）**：接受 WAV 文件路径或一维音频数组

所有方法返回 `(audio_array, sample_rate)` 元组。

### 声码器对比

<div align="center"><img src="Figure/MSPT.png" alt="时频谱对比" width="800px" /></div>

左：原始时频数据。右：经声码器转换后重建的时频谱。

### MusicNet 风格迁移

| 解码器 ID | 0 | 1 | 2 | 3 | 4 | 5 |
|-----------|---|---|---|---|---|---|
| 乐器 | 伴奏小提琴 | 独奏大提琴 | 独奏钢琴 | 独奏钢琴 | 弦乐四重奏 | 管风琴五重奏 |
| 作曲家 | 贝多芬 | 巴赫 | 巴赫 | 贝多芬 | 贝多芬 | 卡姆比尼 |

## 数据与模型

示例数据和预训练模型托管在 [Hugging Face Hub](https://huggingface.co/SukiYume/astrosonify)，首次使用时自动下载。

原始独立脚本和数据文件请查看 [`legacy/original-scripts`](https://github.com/SukiYume/MSP/tree/legacy/original-scripts) 分支。

## 许可

MIT License。见 [LICENSE](./LICENSE)。

第三方模型代码：
- HiFi-GAN：[jik876/hifi-gan](https://github.com/jik876/hifi-gan)（MIT）
- MusicNet：[facebookresearch/music-translation](https://github.com/facebookresearch/music-translation)
