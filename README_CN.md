# RadioSonify

[English](README.md)

RadioSonify 将一维、二维和三维数值数据转换为时长可控的音频。统一的 Python API 与命令行流程覆盖科学数组预处理、声化、可选神经音色转换、WAV 输出和复现元数据。

## 支持的数据

| 输入 | 标准布局 | 默认方法 | 音频 |
|---|---|---|---|
| 一维轮廓 | `(time,)` | `amplitude` | 单声道 |
| 二维矩阵 | `(time, feature)` | `erb` | 单声道 |
| 三维分层矩阵 | `(layer, time, feature)` | `spatial_erb` | 立体声 |

二维输入可表示动态谱、声谱图、图像或其他有序的时间乘特征矩阵。三维输入可表示偏振分量、图像通道、传感器分层或其他并列矩阵。`data_duration` 给出标准时间轴代表的物理时间跨度。

## 安装

RadioSonify 支持 Python 3.9 至 3.13。

```bash
git clone https://github.com/SukiYume/MSP.git
cd MSP
python -m pip install .
```

各神经后端使用对应扩展依赖：

```bash
python -m pip install ".[hifigan]"
python -m pip install ".[musicnet]"
python -m pip install ".[rave]"
python -m pip install ".[all]"
```

## 第一次声化

```python
from pathlib import Path

import radiosonify as rs

data = rs.load_example("raw_burst")
result = rs.sonify(
    data,
    data_duration=2.4,
    method="erb",
    preprocess_params={"scale_statistic": "mad"},
    output=Path("output/burst.wav"),
)

print(result.sample_rate, result.output_duration)
print(result.preprocess_params)
print(result.method_params)
```

对应的命令行为：

```bash
radiosonify download-examples --dest ./data
radiosonify sonify \
  --input data/RawBurst.npy \
  --output output/burst.wav \
  --duration 2.4 \
  --method erb \
  --preprocess scale_statistic=mad
```

CLI 设置使用可重复的 `KEY=VALUE` 选项。数值、元组、字典、布尔值和 `None` 采用 Python 字面量语法，普通单词解析为字符串。

## 输入轴与数据快照

`SonificationInput` 保存标准布局的不可变 `float64` 快照。结果同时记录调用者的原始形状和轴声明。

| 维数 | 默认来源轴 |
|---|---|
| 一维 | 时间轴 `0` |
| 二维 | 时间轴 `0` |
| 三维 | 层轴 `0`，时间轴 `1` |

其他来源布局可显式声明：

```python
import numpy as np

cube = np.load("layers.npy")
source = rs.SonificationInput(
    cube,
    duration=3.0,
    layer_axis=2,
    time_axis=0,
    name="observation-17",
)
result = rs.sonify(source, method="spatial_erb")
```

标准输入域包含有限实数。`nan_policy="propagate"` 将 NaN 视作掩码，并在归一化后映射为静音。复数和无穷值会触发校验错误。

## 处理顺序

每次调用都会在数组变换前解析出一份不可变执行计划。计划阶段统一校验所选方法、全部设置、特征与帧几何、计划层数、输出路径、可选依赖、模型资产和神经模型通道契约。执行阶段采用以下顺序：

```text
标准输入快照
→ 层/时间/特征轴尺寸调整
→ 基线与尺度校正
→ 可选分位裁剪
→ 时间线重复与平滑
→ 归一化到 [0, 1]
→ 主声化
→ 时长拟合
→ 可选音频后处理
→ 输出采样率转换与 WAV 整形
```

各阶段拥有单一数据契约：预处理负责科学数组变换，主方法负责可听映射，后处理器负责音频域风格转换，输出整形负责最终容器。

## 共享预处理

| 设置 | 用途 |
|---|---|
| `layer_rebin` | 三维数据的目标层数，采用有序面积平均 |
| `time_rebin` | 目标时间格数；带帧几何的方法可解析 `"auto"` |
| `feature_rebin` | 目标特征格数 |
| `baseline_operation` | `"subtract"`、`"divide"` 或 `None` |
| `baseline_statistic` | `"median"` 或 `"mean"` |
| `baseline_axis` | 校正轴或 `"auto"` |
| `scale_statistic` | 逐通道 `"mad"`、`"std"` 或 `None` |
| `clip_percentiles` | 全数组 `(lower, upper)` 分位数组合或 `None` |
| `time_smoothing` | 标准时间轴上的高斯 sigma 或 `None` |
| `normalization_scope` | `"global"`、`"per_layer"` 或 `"auto"` |
| `nan_policy` | `"raise"` 或 `"propagate"` |

降采样使用覆盖完整来源范围的等宽面积平均。时间轴与特征轴升采样使用格中心插值。`layer_rebin` 执行降维并保留层顺序。三维数据默认逐层归一化，让每个并列层保持可听；`layer_gains` 可将显式科学权重带入空间合成。

`repeat` 在时间平滑之前沿标准时间轴拼接副本，使重复观测形成连续时间线。

预处理也可以独立调用，便于分析和检查：

```python
import numpy as np

layers = np.load("layers.npy")
prepared_layers = rs.preprocess(
    layers,
    layer_rebin=4,
    time_rebin=2048,
    feature_rebin=512,
    scale_statistic="mad",
)
defaults = rs.preprocessing_defaults()
```

## 主声化方法

| 方法 | 输入 | 映射 | 主要控制项 | 扩展依赖 |
|---|---|---|---|---|
| `profile` | 一维、二维 | 插值轮廓波形与可选解析乐器响应 | `sr`、`instrument` | 基础安装 |
| `amplitude` | 一维、二维 | 谐波载波上的轮廓振幅包络 | `sr`、`freq`、`compression`、`harmonics`、`harmonic_decay` | 基础安装 |
| `erb` | 二维 | 时间映射到时间，有序特征位置映射到感知音高，亮度与时间显著性映射到声级 | 频率范围、频带数、音色、包络、事件和声级设置 | 基础安装 |
| `griffinlim` | 二维 | 类 mel 幅度解释与确定性迭代相位重建 | `sr`、`n_iter`、`n_fft`、`frame_length`、`preemphasis`、`max_db`、`ref_db` | 基础安装 |
| `hifigan` | 二维 | checkpoint 专用 log-mel 适配器与 HiFi-GAN 声码器 | 注册模型几何 | `hifigan` |
| `spatial_erb` | 三维 | 每层独立 ERB 合成与恒功率立体声声像 | ERB 控制项、`pan_positions`、`layer_gains` | 基础安装 |

### ERB 与 spatial ERB

ERB 合成使用重叠感知频带、相位连续载波、起音/释音平滑、受限听觉声级补偿、RMS 控制和真峰值限制。`frequency_scale="mel"` 使用 HTK mel 间距，`frequency_scale="erb"` 使用 ERB-rate 间距。`n_bands` 控制频谱细节，`None` 会根据所选频率范围计算频带数。

可用音色包括 `sine`、`retro_digital`、`warm_pad`、`soft_marimba`、`glass_bell` 和 `instrument_palette`。`instrument_palette` 随音高连续交叉混合互补音色。`event_voice="water_drop"` 根据时间显著性加入确定性瞬态点缀。

```python
matrix = rs.load_example("raw_burst")
result = rs.sonify(
    matrix,
    data_duration=2.4,
    method="erb",
    method_params={
        "min_freq": 90.0,
        "max_freq": 6000.0,
        "n_bands": 48,
        "frequency_scale": "mel",
        "timbre": "instrument_palette",
        "event_voice": "water_drop",
        "voice_params": {"harmonic_limit_hz": 6500.0},
        "event_params": {"max_events_per_second": 3.0, "level_db": -16.0},
    },
)
```

`spatial_erb` 的 `pan_positions` 取值范围为 `-1` 至 `1`，`layer_gains` 取值为 `0` 或正数。两个序列都需要为 `layer_rebin` 之后的每个计划层提供一个值。

### Griffin-Lim 与 HiFi-GAN

Griffin-Lim 根据 `n_fft`、`frame_length`、采样率、时长和重复次数推导有效特征格数与自动时间格数。HiFi-GAN 接收共享归一化矩阵，并在模型适配器内执行已发布 checkpoint 的固定 80 格编码。随数据变化的直方图偏移记录在 `result.method_params` 中。

## 音频后处理器

`musicnet` 将单声道主音频转换为六种预训练 WaveNet 音乐风格之一，原生采样率为 16 kHz。编码器需要重采样后至少 800 个样本，对应 50 ms 的主音频。计划阶段会在科学预处理之前验证长度，并解析依赖和固定模型资产。

`rave` 使用用户提供的 TorchScript 导出模型。计划阶段先在 CPU 加载模型，读取标准 nn~ `sampling_rate` 与 `forward_params` 元数据，解析输入/输出采样率，并在科学预处理之前验证通道兼容性。推理阶段在所选 `cpu`、`cuda` 或 `mps` 设备重新核对同一契约。单进单出的单声道模型可逐声道处理立体声，单声道来源也可扩展到模型的多通道输入。

RAVE 导出文件适合来自可信来源，因为 `torch.jit.load` 会执行模型代码。生成音频时可同时记录模型来源和许可证。

```python
profile = rs.load_example("profile")
matrix = rs.load_example("raw_burst")
music_style = rs.sonify(
    profile,
    data_duration=2.0,
    method="amplitude",
    postprocess="musicnet",
    postprocess_params={"decoder_id": 2, "seed": 0},
)

rave_style = rs.sonify(
    matrix,
    data_duration=2.0,
    method="erb",
    postprocess="rave",
    postprocess_params={
        "model_path": "/path/to/trusted-model.ts",
        "device": "auto",
        "seed": 0,
    },
)
```

## 时长与输出

全部方法采用同一个时长公式：

```text
target_duration = data_duration × repeat ÷ speed
```

`speed=2` 产生来源时长的一半，`speed=0.5` 产生来源时长的两倍。`amplitude` 的注册重复默认值为 `5`，其余主方法为 `1`。显式 `repeat` 会覆盖注册值。

`preserve_pitch=True` 选择相位声码器时间伸缩。标准多相路径会让播放速度与音高一起变化。`output_sr` 选择最终采样率，并保持时长和物理音高。提供 `output_sr` 时，最终样本数为 `round(output_sr × target_duration)`。

输出整形会移除直流分量、添加短边缘淡入淡出、约束峰值、创建父目录，并在路径校验后写入 WAV。

## 结果与复现

`sonify` 返回不可变的 `SonificationResult`。

| 字段 | 内容 |
|---|---|
| `audio`、`sample_rate`、`output_duration`、`output_path` | 最终波形与容器 |
| `data_type`、`data_duration`、`input_shape`、`source_name` | 来源标识与物理范围 |
| `source_time_axis`、`source_layer_axis` | 调用者原始布局中的轴 |
| `method`、`preprocess_params`、`method_params` | 主映射与完整解析设置 |
| `speed`、`repeat`、`preserve_pitch`、`target_duration` | 时长契约 |
| `method_sample_rate`、`method_native_samples`、`method_native_duration`、`method_time_scale` | 主合成时序 |
| `postprocess`、`postprocess_params`、`postprocess_native_samples`、`postprocess_native_duration`、`postprocess_time_scale` | 可选音频风格阶段 |

参数映射会递归复制并冻结，数值数组使用不可变字节缓冲区快照。用于发表和复现时，可记录 RadioSonify 版本、来源校验和、来源轴、物理时长、结果参数映射、模型 revision 和输出采样率。

## 功能查询与示例资产

```bash
radiosonify list-methods
radiosonify list-settings
radiosonify --help
radiosonify download-examples --dest ./data
```

```python
rs.available_methods()
rs.available_methods("matrix")
rs.available_postprocessors()
rs.default_method("matrix")

profile = rs.load_example("profile")
burst = rs.load_example("burst")
raw_burst = rs.load_example("raw_burst")
parkes_burst = rs.load_example("parkes_burst")
```

示例数组、HiFi-GAN 权重和 MusicNet checkpoint 来自固定 revision 的 [`TorchLight/radiosonify`](https://huggingface.co/TorchLight/radiosonify)，revision 存储在 `radiosonify.hub.REVISION`。资产在首次使用时下载到 `~/.cache/radiosonify`，`RADIOSONIFY_CACHE_DIR` 可选择其他缓存目录。`profile` 的解析乐器响应在本地生成，并与下载资产共用缓存目录。[MODEL_ASSETS.md](MODEL_ASSETS.md) 记录来源、转换、核验历史和许可证范围。

## 开发与许可证

[CONTRIBUTING.md](CONTRIBUTING.md) 包含开发环境、模块边界和验证命令。[CHANGELOG.md](CHANGELOG.md) 记录各版本变更。

MSP 自有代码使用 [MIT License](LICENSE)。内置 MusicNet 推理子集与 checkpoint 使用带非商业用途条件的 CC BY-NC 4.0。分发元数据采用 `MIT AND CC-BY-NC-4.0`，[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 提供各组件条款。
