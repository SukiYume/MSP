<div align="center">

<div align="center"><img style="border-radius:50%;border: royalblue dashed 1px;padding: 5px" src="assets/Burst.png" alt="RMS" width="140px" /></div>

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

## 更新内容（v0.1.1）

- 修复 `rebin_spectrogram()` 在目标 bin 超过输入尺寸时的崩溃问题。
- 修正 `amplitude_modulate(freq=...)` 的频率语义，使 `freq` 直接对应物理 Hz。
- 在 `musicnet()` 中增加 CUDA 可用性显式检查。
- 模型加载在支持时使用更安全的 `torch.load(..., weights_only=True)`。
- CLI 新增 `astrosonify astronify`（方法 0）子命令。
- 新增缓存目录环境变量 `ASTROSONIFY_CACHE_DIR`。

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

### 开发环境（可复现测试最短路径）

```bash
python -m pip install --upgrade pip
pip install -e .[dev]
pytest -q
```

说明：
- 在部分平台上，`soundfile` 可能需要系统层 `libsndfile` 依赖。
- 可选方法依赖默认拆分安装：`astronify`、`hifigan`、`musicnet`。

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

# 方法 5：WaveNet 音乐风格迁移（需要 torch，推荐 CUDA）
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

# 使用 astronify 音高映射声化
astrosonify astronify --input profile.npy --output astronify.wav --note-spacing 0.02 --downsample 5

# 使用振幅调制声化
astrosonify amplitude --input profile.npy --output amp.wav --freq 1000

# 下载示例数据
astrosonify download-examples --dest ./data/
```

现在 CLI 中所有命令的 `--input` 都会进行路径存在性校验，错误提示更友好。

## 方法列表

| # | 方法 | 函数 | 额外依赖 |
|---|------|------|---------|
| 0 | Astronify（音高映射）| `astronify_sonify()` | astropy, astronify |
| 1 | 脉冲轮廓转波形 | `profile_to_wave()` | 无 |
| 2 | 振幅调制 | `amplitude_modulate()` | 无 |
| 3 | Griffin-Lim 声码器 | `griffinlim()` | 无 |
| 4 | HiFi-GAN 神经声码器 | `hifigan()` | torch, scikit-image |
| 5 | WaveNet 风格迁移 | `musicnet()` | torch, tqdm（CUDA 可选但更快） |

### 输入处理

- **轮廓类方法（0, 1, 2）**：接受一维轮廓或二维时频谱（自动沿频率轴平均）
- **时频谱方法（3, 4）**：接受二维时频谱（自动重采样到目标维度）
- **MusicNet（5）**：接受 WAV 文件路径或一维音频数组

所有方法返回 `(audio_array, sample_rate)` 元组。

### 输入长度与下采样参考

设二维输入形状为 `(T, F)`，一维输入长度为 `N`。

| # | 方法 | 输出采样率 | 输入点数与输出时长关系 | 建议下采样目标 |
|---|------|-----------|------------------------|---------------|
| 0 | `astronify_sonify` | 由 astronify 输出 WAV 决定 | 若有效点数 `L = floor(T / d)`（或 `floor(N / d)`），则时长约为 `L * note_spacing` 秒（默认 `note_spacing=0.01`） | 建议 `L` 取 200-2000（例如 `d = T / 1000`） |
| 1 | `profile_to_wave` | 用户指定 `sr`（默认 48000） | 输出时长严格等于 `duration`（默认 10 秒），插值后与输入点数无直接关系 | 为保证包络细节，建议有效长度 `L` 取 200-5000 |
| 2 | `amplitude_modulate` | 用户指定 `sr`（默认 48000） | 输出时长严格等于 `duration`（默认 2 秒），插值后与输入点数无直接关系 | 与方法 1 类似，建议 `L` 取 200-5000 |
| 3 | `griffinlim` | 用户指定 `sr`（默认 48000） | 设 `time_rebin=B_t`，时长约为 `B_t * (frame_length/4)`；默认 `frame_length=0.04`，即约 `B_t * 0.01` 秒 | 建议 `time_rebin ≈ 100 * 目标秒数`；`freq_rebin` 取 256-512 |
| 4 | `hifigan` | 来自模型配置 `sampling_rate`（当前模型 22050） | 设 `time_rebin=B_t`，时长约为 `B_t * hop_size / sampling_rate`；当前模型 `hop_size=256`，即约 `B_t * 0.01161` 秒 | 建议 `time_rebin ≈ 目标秒数 * 22050 / 256`（约 `86 * 目标秒数`） |
| 5 | `musicnet` | 用户指定 `sr`（默认 48000） | 基本保持输入 WAV 时长，但受模型步长量化：输出采样点约 `floor(N/800) * 800`（当前 `encoder_pool=800`） | 一般无需额外下采样；超长音频建议先切片再转换 |

#### 与 legacy 脚本一致的固定参数（参考）

如果你希望和 legacy 脚本保持一致，测试中推荐固定使用：

- 方法 3（`griffinlim`）：`time_rebin=128`, `freq_rebin=512`
- 方法 4（`hifigan`）：`time_rebin=128`（频率轴在模型流程中固定重采样到 80 mel bins）

这些固定值已经用于测试中的可复现实例；实际生产使用仍建议按目标时长做参数化配置。

#### 二维数据的另一维（频率轴）应取多大

- 方法 0/1/2：二维输入会先沿频率轴做平均（`mean(axis=1)`），因此没有硬性频率点数要求；建议至少保留 `F >= 32`，常见可用范围 `64-1024`。
- 方法 3（`griffinlim`）：频率轴会重采样到 `freq_rebin`（若不设则用 `n_mels`，默认 `512`）；建议输出频率点数取 `256-512`，legacy 对齐可用 `512`。
- 方法 4（`hifigan`）：模型前处理会强制把频率轴缩放到 `80` 个 mel bins；输入频率轴无需固定，但建议原始 `F >= 80`（常见 `256-1024`）以减少频率细节损失。

可按下面思路设置：先根据目标时长确定 `time_rebin`，再把方法 3 的 `freq_rebin` 设为 `256` 或 `512`；方法 4 只需保证原始频率分辨率不要过低。

#### 常见输入形状的推荐参数速查表

> 下面以“语音长度约 1-3 秒、保留主要结构且可听”为目标给出经验值。

| 输入形状 `(T, F)` | 方法 3 (`griffinlim`) 推荐参数 | 方法 4 (`hifigan`) 推荐参数 | 预估时长 |
|---|---|---|---|
| `(1024, 256)` | `time_rebin=100`, `freq_rebin=256` | `time_rebin=100` | M3: ~1.0s；M4: ~1.16s |
| `(2048, 512)` | `time_rebin=128`, `freq_rebin=512` | `time_rebin=128` | M3: ~1.28s；M4: ~1.49s |
| `(4096, 512)` | `time_rebin=200`, `freq_rebin=512` | `time_rebin=200` | M3: ~2.0s；M4: ~2.32s |
| `(8192, 1024)` | `time_rebin=300`, `freq_rebin=512` | `time_rebin=300` | M3: ~3.0s；M4: ~3.48s |

说明：

- 方法 3 时长近似 `time_rebin × 0.01s`（默认参数下）。
- 方法 4 时长近似 `time_rebin × 256 / 22050 ≈ time_rebin × 0.01161s`。
- 若你更关注频率细节，可优先把方法 3 的 `freq_rebin` 设为 `512`；若更关注速度和轻量，可设为 `256`。

### 声码器对比

<div align="center"><img src="assets/MSPT.png" alt="时频谱对比" width="800px" /></div>

左：原始时频数据。右：经声码器转换后重建的时频谱。

### MusicNet 风格迁移

| 解码器 ID | 0 | 1 | 2 | 3 | 4 | 5 |
|-----------|---|---|---|---|---|---|
| 乐器 | 伴奏小提琴 | 独奏大提琴 | 独奏钢琴 | 独奏钢琴 | 弦乐四重奏 | 管风琴五重奏 |
| 作曲家 | 贝多芬 | 巴赫 | 巴赫 | 贝多芬 | 贝多芬 | 卡姆比尼 |

## 数据与模型

示例数据和预训练模型托管在 [Hugging Face Hub](https://huggingface.co/TorchLight/astrosonify)，首次使用时自动下载。

### 缓存目录

默认缓存目录是 `~/.cache/astrosonify`。

你可以通过环境变量覆盖：

```bash
export ASTROSONIFY_CACHE_DIR=/path/to/cache
```

Windows PowerShell：

```powershell
$env:ASTROSONIFY_CACHE_DIR = "D:\\astrosonify-cache"
```

### 模型加载安全说明

本项目从 Hugging Face Hub 官方仓库下载模型权重。加载时在可用版本上优先使用更安全的 weights-only 路径（`torch.load(..., weights_only=True)`），并对旧版 PyTorch 保持兼容回退。

原始独立脚本和数据文件请查看 [`legacy/original-scripts`](https://github.com/SukiYume/MSP/tree/legacy/original-scripts) 分支。

## 许可

MIT License。见 [LICENSE](./LICENSE)。

第三方模型代码：
- HiFi-GAN：[jik876/hifi-gan](https://github.com/jik876/hifi-gan)（MIT）
- MusicNet：[facebookresearch/music-translation](https://github.com/facebookresearch/music-translation)
