# RadioSonify

[English](README.md)

RadioSonify 将一维、二维和三维数值数组转换为时长可控的音频。项目提供统一 Python API、命令行入口、确定性信号处理方法、可选神经音色变换，以及完整的结果溯源信息。

## 项目概览

| 输入 | 标准布局 | 默认方法 | 输出 |
|---|---|---|---|
| 一维轮廓 | `(time,)` | `amplitude` | 单声道 |
| 二维矩阵 | `(time, feature)` | `erb` | 单声道 |
| 三维分层矩阵 | `(layer, time, feature)` | `spatial_erb` | 立体声 |

输入通过 `data_duration` 声明其物理时间跨度。RadioSonify 依次执行共享预处理、方法选择与校验、目标时长拟合、波形整形，并可写出 WAV 文件。

## 安装

RadioSonify 支持 Python 3.9 至 3.13。源码安装命令如下：

```bash
git clone https://github.com/SukiYume/MSP.git
cd MSP
python -m pip install .
```

神经后端按所需方法安装：

```bash
python -m pip install ".[hifigan]"
python -m pip install ".[musicnet]"
python -m pip install ".[rave]"
python -m pip install ".[all]"
```

## 示例数据与本地缓存

首次运行可直接使用四份示例数组：

```python
import radiosonify as rs

profile = rs.load_example("profile")             # 一维
burst = rs.load_example("burst")                 # 二维，已校正
raw_burst = rs.load_example("raw_burst")         # 二维，原始记录
parkes_burst = rs.load_example("parkes_burst")   # 二维，原始记录
```

同样的数组也可保存为文件：

```bash
radiosonify download-examples --dest ./data
```

示例数组、HiFi-GAN 权重和 MusicNet checkpoint 均来自固定 revision 的 [`TorchLight/radiosonify`](https://huggingface.co/TorchLight/radiosonify)，该 revision 记录在 `radiosonify.hub.REVISION`，每个文件在首次使用时下载。下载内容保存在 `~/.cache/radiosonify`；在导入本包之前设置 `RADIOSONIFY_CACHE_DIR` 可选择其他目录。`profile` 方法的乐器响应在本地合成，并缓存到同一目录。[MODEL_ASSETS.md](MODEL_ASSETS.md) 记录资产来源、转换过程、核验历史和许可证范围。

## 快速开始

### Python

```python
from pathlib import Path

import numpy as np
import radiosonify as rs

profile_result = rs.sonify(
    rs.load_example("profile"),
    data_duration=1.2,
    method="amplitude",
    repeat=1,
)

matrix_result = rs.sonify(
    rs.load_example("raw_burst"),
    data_duration=2.4,
    method="erb",
    preprocess_params={"scale_statistic": "mad"},
    output=Path("output/matrix.wav"),
)

spatial_result = rs.sonify(
    np.load("layers.npy"),
    data_duration=3.0,
    method="spatial_erb",
)
```

每个结果包含冻结音频快照、采样率、所选方法、来源几何、时序数值、有效预处理设置、方法设置、后处理设置和输出路径。

### 命令行

```bash
radiosonify sonify \
  --input matrix.npy \
  --output output/matrix.wav \
  --duration 2.4 \
  --method erb \
  --preprocess scale_statistic=mad
```

CLI 设置采用可重复的 `KEY=VALUE` 选项。数值、元组、布尔值和 `None` 使用 Python 字面量语法，普通单词按字符串解析。

### 功能查询

两个界面都可以列出当前安装版本的参数面：

```bash
radiosonify list-methods
radiosonify list-settings
radiosonify --help
```

```python
rs.available_methods()           # 全部已注册主方法
rs.available_methods("matrix")   # 接受某一输入类型的方法
rs.available_postprocessors()
rs.default_method("matrix")
```

## 输入契约

`SonificationInput` 将调用者数据复制为标准布局的 `float64` 数组，并采用冻结底层缓冲区。调用者后续修改原数组时，存储快照保持构造时的数值；NumPy 写标记持续保持关闭状态。

默认轴规则如下：

| 维数 | 含义 | 默认轴 |
|---|---|---|
| 一维 | 时间轮廓 | 时间轴 `0` |
| 二维 | 时间乘特征矩阵 | 时间轴 `0` |
| 三维 | 多个时间乘特征分层 | 来源层轴 `0`，标准化后的时间轴 `1` |

其他来源布局通过轴参数声明：

```python
source = rs.SonificationInput(
    data,
    duration=2.0,
    time_axis=1,
    layer_axis=2,
    name="observation-17",
)
result = rs.sonify(source, method="spatial_erb")
```

标准输入域为有限实数。`nan_policy="propagate"` 将 NaN 视为掩码样本，并在归一化后映射为静音。无穷值和复数会触发输入校验错误。

## 共享预处理

所有主声化方法接收归一化到 `[0, 1]` 的数组。流水线顺序如下：

```text
时间轴与特征轴重分箱
→ 基线校正
→ 逐通道尺度校正
→ 分位裁剪
→ 重复
→ 时间平滑
→ min-max 归一化
```

| 设置 | 用途 |
|---|---|
| `time_rebin` | 目标时间格数；带帧几何的方法可使用 `"auto"` |
| `feature_rebin` | 目标特征格数 |
| `baseline_operation` | `"subtract"`、`"divide"` 或 `None` |
| `baseline_statistic` | `"median"` 或 `"mean"` |
| `baseline_axis` | 基线统计与 `scale_statistic` 共用的来源轴，或 `"auto"` |
| `scale_statistic` | 逐通道 `"mad"`、`"std"` 或 `None` |
| `clip_percentiles` | 在整个数组上统计的 `(lower, upper)` 分位数组合，或 `None` |
| `time_smoothing` | 时间轴高斯宽度或 `None` |
| `normalization_scope` | `"global"`、`"per_layer"` 或 `"auto"` |
| `nan_policy` | `"raise"` 或 `"propagate"` |

`repeat` 属于统一时序契约，并沿标准时间轴连接数据副本。时间平滑覆盖连接边界。

该阶段也可单独调用，方法专用 CLI 命令在分发之前执行的正是同一阶段：

```python
normalized = rs.preprocess(raw, scale_statistic="mad", time_rebin=2048)
rs.preprocessing_defaults()
```

## 声化方法

| 方法 | 输入 | 映射 | 主要控制 | Extra |
|---|---|---|---|---|
| `profile` | 一维、二维 | 插值轮廓波形，可叠加解析生成的乐器响应 | `sr`、`instrument` | base |
| `amplitude` | 一维、二维 | 轮廓振幅包络控制谐波载波 | `sr`、`freq`、`compression`、`harmonics`、`harmonic_decay` | base |
| `erb` | 二维 | 时间映射到时间，有序特征映射到感知音高，亮度和时间显著性映射到振幅 | 频率、对比度、音色、包络、事件和电平设置 | base |
| `griffinlim` | 二维 | 以 mel 风格幅度解释矩阵，通过确定性迭代相位重建音频 | `sr`、`n_iter`、`n_fft`、`frame_length`、`preemphasis`、`max_db`、`ref_db` | base |
| `hifigan` | 二维 | 固定 checkpoint 适配器和 HiFi-GAN 声码器 | 注册模型几何 | `hifigan` |
| `spatial_erb` | 三维 | 每层执行一次 ERB 合成，并采用恒功率立体声声像 | ERB 设置、`pan_positions`、`layer_gains` | base |

ERB 方法采用重叠归一化感知频带、连续相位载波、确定性音色、可选事件重音、包络平滑、听觉电平补偿、RMS 控制和真峰值限制。

传入 `pan_positions` 和 `layer_gains` 时，使用每层各含一个有限值的可重复读取序列；声像值范围为 `-1` 至 `1`，增益为 `0` 或更大值。

`timbre` 在 `sine`、`retro_digital`、`warm_pad`、`soft_marimba`、`glass_bell` 和 `instrument_palette` 中选择载波波形。`event_voice` 可选 `none` 或 `water_drop`。进阶波形与事件控制归入 `voice_params` 和 `event_params` 两个映射，两种 ERB 方法共用：

| 映射 | 键 | 默认值 | 取值范围 |
|---|---|---|---|
| `voice_params` | `harmonic_limit_hz` | `3500.0` | 大于 `0`，上限为 `0.475 × sr` |
| `voice_params` | `detune_cents` | `10.0` | `0` 至 `50` |
| `voice_params` | `fm_index` | `1.0` | `0` 至 `1` |
| `voice_params` | `chorus_rate_hz` | `0.45` | `0` 至 `10` |
| `voice_params` | `chorus_depth_ms` | `8.0` | `0` 至 `20` |
| `event_params` | `salience_threshold` | `0.35` | `0` 至 `1` |
| `event_params` | `max_events_per_second` | `6.0` | `0` 至 `100` |
| `event_params` | `decay_ms` | `70.0` | `1` 至 `5000` |
| `event_params` | `level_db` | `-20.0` | `0` 及以下 |

`harmonic_limit_hz` 约束全部音色的泛音上限。`detune_cents`、`chorus_rate_hz` 和 `chorus_depth_ms` 作用于 `retro_digital`、`warm_pad` 和 `instrument_palette`；`fm_index` 作用于 `retro_digital`。`event_params` 各键在选择 `event_voice="water_drop"` 后生效。

```python
result = rs.sonify(
    matrix,
    data_duration=2.4,
    method="erb",
    method_params={
        "timbre": "glass_bell",
        "event_voice": "water_drop",
        "voice_params": {"harmonic_limit_hz": 6000.0},
        "event_params": {"max_events_per_second": 3.0, "level_db": -14.0},
    },
)
```

Griffin-Lim 根据 FFT 设置解析时间与特征几何。HiFi-GAN 接收共享归一化矩阵，并在方法内部执行 checkpoint 所需的 80 格特征编码。

## 可选后处理

`musicnet` 在 16 kHz 下应用六种预训练 WaveNet 音乐风格。RadioSonify 会在主声化前校验依赖和固定模型资产，补齐编码器末尾窗口，并将解码音频裁回精确输入跨度。

`rave` 应用用户提供的可信 TorchScript 模型，并读取模型采样率和声道元数据。RAVE TorchScript 加载会执行模型代码，因此每个 RAVE 导出文件都应来自可信来源。

```python
styled = rs.sonify(
    profile,
    data_duration=2.0,
    method="amplitude",
    postprocess="musicnet",
    postprocess_params={"decoder_id": 2, "seed": 0},
)
```

## 时长与输出

所有输入维数、主方法和后处理器共享同一目标时长公式：

```text
target_duration = data_duration × repeat ÷ speed
```

`speed=2` 生成一半时长，`speed=0.5` 生成两倍时长。`amplitude` 的注册重复默认值为 `5`，其他主方法为 `1`。显式 `repeat` 可控制所有方法。

`preserve_pitch=True` 使用相位声码器伸缩。标准多相路径同步改变播放速度和音高。`output_sr` 转换最终容器采样率，同时保持物理时长与音高。最终样本数为 `round(sample_rate * target_duration)`。

输出整形会移除直流分量、添加短边缘淡化，并将波形限制在 WAV 范围内。保存流程先校验路径，再创建父目录并写入 WAV 文件。

## 结果与溯源

`sonify` 返回冻结的 `SonificationResult`。音频数组和数值数组元数据均使用冻结底层缓冲区。

| 字段 | 含义 |
|---|---|
| `audio`、`sample_rate`、`output_duration`、`output_path` | 最终波形和容器信息 |
| `data_type`、`data_duration`、`input_shape`、`source_name` | 来源身份和物理跨度 |
| `source_time_axis`、`source_layer_axis` | 调用者原始布局中的解析轴 |
| `method`、`preprocess_params`、`method_params` | 所选映射和有效设置 |
| `speed`、`repeat`、`preserve_pitch`、`target_duration` | 时序控制 |
| `method_sample_rate`、`method_native_samples`、`method_native_duration`、`method_time_scale` | 主声化阶段时序 |
| `postprocess`、`postprocess_params`、`postprocess_native_samples`、`postprocess_native_duration`、`postprocess_time_scale` | 可选风格阶段时序和设置 |

参数映射会递归复制并冻结。HiFi-GAN 还会在 `method_params` 中记录数据相关的 `histogram_offset`。

## 底层 API 与 CLI 适配器

底层方法函数适用于自行管理预处理和时序的实验：

```python
rs.profile_to_wave(...)
rs.amplitude_modulate(...)
rs.erb_sonify(...)
rs.griffinlim_reconstruct(...)
rs.hifigan_vocode(...)
rs.spatial_sonify(...)
rs.musicnet_transform(...)
rs.rave_transform(...)
```

方法专用 CLI 适配器包括 `profile`、`amplitude`、`erb`、`spatial-erb`、`griffinlim`、`hifigan`、`musicnet` 和 `rave`。Griffin-Lim 接受已弃用的 `--n-mels`、`--freq-rebin` 和 `--time-rebin` 别名，并将其路由到共享预处理。新脚本可直接使用 `--preprocess`。[CHANGELOG.md](CHANGELOG.md) 记录公开迁移和弃用计划。

## 科学使用

声化是数值结构的解释性表达。基线校正、裁剪、归一化、重采样、合成和神经风格变换都会影响听觉结果。可复现实验应保存源数据，并记录 RadioSonify 版本、输入校验和、物理时长、轴声明、有效参数、模型 revision 和输出采样率。

`profile`、`amplitude`、`erb` 和 `spatial_erb` 提供确定性信号处理映射。Griffin-Lim 使用确定性相位初值。MusicNet 支持记录随机种子。RAVE 行为由所提供模型决定。

## 开发与许可证

[`CONTRIBUTING.md`](CONTRIBUTING.md) 提供开发环境、验证命令和贡献规则。

MSP 编写的代码采用 [MIT License](LICENSE)。随附的 MusicNet 推理子集和 checkpoint 采用 CC BY-NC 4.0，并带有非商业用途条件。发行元数据采用 `MIT AND CC-BY-NC-4.0`。[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 提供各随附组件的组件级条款。
