<h1 align="center">MSP · RadioSonify</h1>

<p align="center">
  <img src="https://raw.githubusercontent.com/SukiYume/MSP/main/assets/Burst.png" alt="射电脉冲可视化" width="180">
</p>

<p align="center">
  <strong>把射电脉冲数据变成可以听的声音</strong><br>
  输入一条轮廓或一张动态谱，输出一个你指定时长的 WAV。
</p>

<p align="center">
  <img alt="Python 3.9+" src="https://img.shields.io/badge/Python-3.9%2B-3776ab?logo=python&logoColor=white">
  <a href="https://huggingface.co/TorchLight/radiosonify"><img alt="模型与数据" src="https://img.shields.io/badge/Models%20%26%20Data-Hugging%20Face-ffd21e"></a>
  <a href="https://github.com/SukiYume/MSP/blob/main/THIRD_PARTY_NOTICES.md"><img alt="混合许可证" src="https://img.shields.io/badge/License-MIT%20%2B%20CC--BY--NC--4.0-orange"></a>
</p>

<p align="center">
  <a href="#安装">安装</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#你的数据">数据</a> ·
  <a href="#方法">方法</a> ·
  <a href="#时长速度与重复">时长</a> ·
  <a href="#返回结果">结果</a> ·
  <a href="https://github.com/SukiYume/MSP/blob/main/README.md">English</a>
</p>

---

## MSP 做什么

射电望远镜记录的是科学数组。MSP 把其中两种数组映射成声音：

- 一维**脉冲轮廓**，以及
- 二维**动态谱**（时间 × 频率）。

你提供数组和它覆盖的物理时长，MSP 选择合适的方法，按你要求的时长合成音频，并把
用到的全部参数随波形一起返回。

```mermaid
flowchart LR
    A["轮廓 · 一维"] --> C["SonificationInput"]
    B["动态谱 · 时间 × 频率"] --> C
    C --> D["选择方法与参数"]
    D --> E["拟合 时长 × repeat ÷ speed"]
    E --> F["WAV + 可复现元数据"]
    F -. 可选 .-> G["MusicNet 风格化"]
```

## 安装

从 PyPI 安装发布版本：

```bash
python -m pip install radiosonify
```

`profile`、`amplitude` 和 `griffinlim` 三个方法在基础安装下即可运行。需要神经后端时再装：

```bash
python -m pip install "radiosonify[hifigan]"
python -m pip install "radiosonify[musicnet]"
python -m pip install "radiosonify[all]"
```

请安装与你的 CPU 或 CUDA 环境匹配的 PyTorch 版本。

要做可编辑的源码安装，先克隆仓库，再运行 `python -m pip install -e .`；需要全部开发
依赖时使用 `python -m pip install -e ".[all,dev]"`。

示例数组和预训练权重会在首次使用时从
[`TorchLight/radiosonify`](https://huggingface.co/TorchLight/radiosonify) 下载，
固定在同一个 revision 上，缓存于 `~/.cache/radiosonify`。导入前设置
`RADIOSONIFY_CACHE_DIR` 可换到其他位置。
乐器响应在本机由确定性解析波形生成，不再下载任何声音录音。资产来源和许可证见
[MODEL_ASSETS.md](https://github.com/SukiYume/MSP/blob/main/MODEL_ASSETS.md)。

## 快速开始

在源码检出目录中运行内置示例，完整走一遍流程：

```bash
python examples/sonify_example.py
```

### 命令行兼容

0.1.x 的 `radiosonify` 命令继续保留：

```bash
radiosonify list-methods
radiosonify amplitude --input profile.npy --output profile.wav --repeat 5
radiosonify griffinlim --input spectrum.npy --output spectrum.wav
radiosonify download-examples --dest ./data
```

使用 `radiosonify COMMAND --help` 查看完整参数。需要时长感知和完整结果元数据时，推荐
使用 Python 的 `sonify()` API。

### 一条脉冲轮廓

```python
import numpy as np
import radiosonify as rs

profile = np.load("profile.npy", allow_pickle=False)

result = rs.sonify(
    profile,
    data_duration=0.725,       # 数据覆盖的物理时长，单位秒
    method="auto",
    repeat=5,                  # 把数据连续播放 5 次
    method_params={"freq": 880},
    output="profile.wav",
)

print(result.method, result.output_duration, result.sample_rate)
```

### 一张动态谱

```python
from pathlib import Path

import numpy as np
import radiosonify as rs

dynamic_spectrum = np.load("observation.npy", allow_pickle=False)

source = rs.SonificationInput(
    dynamic_spectrum,
    duration=4.2,
    data_type="dynamic_spectrum",   # 省略时按维数推断
    name="candidate-01",
)

result = rs.sonify(
    source,
    method="griffinlim",
    speed=2.0,                      # 2 倍速得到 2.1 秒音频
    method_params={"n_iter": 32, "time_rebin": 256, "freq_rebin": 256},
    output=Path("audio") / "candidate-01.wav",
)
```

## 你的数据

| 类型 | 形状 | 坐标轴含义 |
|---|---:|---|
| `profile` | `(time,)` | 每个相位或时间 bin 一个强度值 |
| `dynamic_spectrum` | `(time, frequency)` | 行沿时间推进，列沿频率推进 |

MSP 接受实数、有限、非空的数组。一维输入读作轮廓，二维输入读作动态谱，因此当形状
本身已经说明数据类型时，`data_type` 可以省略。

数组形状本身不携带时间定标，所以 `data_duration` 是必填的。声化一段更长观测中的
切片时，请传入该切片的时长。

`SonificationInput` 会复制数组并把副本设为只读，让转换过程中的数据保持稳定。

## 方法

`method="auto"` 会按输入类型选择依赖最轻的默认方法：

| 输入 | 默认方法 | 可用方法 |
|---|---|---|
| 轮廓 | `amplitude` | `profile`、`amplitude` |
| 动态谱 | `griffinlim` | `profile`、`amplitude`、`griffinlim`、`hifigan` |

| 方法 | 听起来是什么 | 承载的信息 | 额外依赖 |
|---|---|---|---|
| `profile` | 轮廓形状直接成为波形，可选用小提琴或钢琴采样染色 | 脉冲的时间位置、宽度和相对形状 | — |
| `amplitude` | 轮廓控制一个稳定正弦音的响度 | 脉冲强弱和时间包络 | — |
| `griffinlim` | 完整二维强度图读作幅度谱，由 Griffin–Lim 估计相位 | 时频演化，包括扫描和频带结构 | — |
| `hifigan` | 二维图送入预训练神经声码器，得到更连续的音质 | 时频演化 | `hifigan` |

`profile` 和 `amplitude` 会把动态谱沿频率求均值，再用得到的时间轮廓工作；
`griffinlim` 和 `hifigan` 使用完整的二维结构。

方法自己的设置放在 `method_params`。用注册表查询某个方法接受的完整参数列表：

```python
for method in rs.available_methods("dynamic_spectrum"):
    print(method.name, method.parameters, method.optional_extra)

for postprocessor in rs.available_postprocessors():
    print(postprocessor.name, postprocessor.parameters, postprocessor.optional_extra)
```

### 几个值得了解的设置

`time_rebin` 和 `freq_rebin` 指定目标 bin 数。MSP 在完整坐标轴上做等宽面积平均，
目标尺寸取任意值时每个输入样本都会参与。轮廓方法的 `time_downsample` 起同样的作用。

`compression` 决定 `amplitude` 方法把轮廓强度映射成响度的曲线，公式为
`log1p(compression * x) / log1p(compression)`。默认 `compression=99` 把峰值 1%
的结构提升到包络峰值约 15%；设为 `0` 得到线性包络。

`clean=True` 在合成前做基于百分位的清理，当带通形状或窄带干扰主导强度量程时很有用。

HiFi-GAN 的 `time_smoothing=<sigma>` 以输入时间 bin 为单位沿时间轴平滑，同时保留
各频率通道上持续存在的结构。

Griffin–Lim 默认迭代 64 次估计相位。mel 到线性的近似变换本身存在误差下限，因此在
调高 `n_iter` 前建议先对自己的数据实测收益。

## 时长、速度与重复

所有方法遵循同一条规则：

```text
目标音频时长 = 数据物理时长 × repeat ÷ speed
目标样本数   = round(采样率 × 目标音频时长)
```

`speed=1` 配合 `repeat=1` 保持真实物理时长。`speed=2` 以两倍速播放，`speed=0.5`
以半速播放，`speed=0.1` 可以把毫秒量级的爆发拉长到适合聆听的长度。`repeat=5` 把
数据连续播放五次；由于 MSP 按分箱数据处理轮廓，相邻两次之间衔接平滑。`repeat`
适用于 `profile` 和 `amplitude` 方法。

轮廓映射和振幅调制直接按目标长度合成。Griffin–Lim 和 HiFi-GAN 先产生方法原生波形，
再由时长层重采样到目标长度。这种重采样等效于改变播放速率，音高随时长一同变化；设置
`preserve_pitch=True` 可改用相位声码器做时间拉伸。

对二维方法，`time_rebin` 决定方法原生长度，因而影响最终音高。`SonificationResult`
记录了 `method_native_samples`、`method_native_duration` 和 `method_time_scale`
（拟合后样本数 / 原生样本数），让这层关系保持可见。

各方法有各自的原生采样率：可配置方法为 48 kHz，HiFi-GAN 为 22.05 kHz，经 MusicNet
后为 16 kHz。需要一批文件共用同一容器采样率时，传入 `output_sr=48_000`。该转换保持
时长与音高，可听带宽维持在方法的原生上限。

统一流程的每个输出最后都会去直流、施加最长 5 毫秒的边缘淡化、把峰值归一化到 `0.9`，
并保持精确的目标样本数。

## 可选的 MusicNet 风格化

MusicNet 以音频为输入，因此作为后处理器接在主方法之后：

```python
result = rs.sonify(
    source,
    method="amplitude",
    postprocess="musicnet",
    postprocess_params={"decoder_id": 2, "seed": 0},
    output_sr=48_000,
    output="styled.wav",
)
```

共有六种风格解码器，见 `radiosonify.musicnet.STYLE_NAMES`。生成过程是随机的，默认
`seed=0` 让重复运行结果一致，同时保持调用者的全局 PyTorch 随机状态不变；传入
`seed=None` 可每次重新采样。MusicNet 在其原生 16 kHz、正常播放速度下运行，`speed`
由 MSP 在之后施加。较长的输入分段解码，段与段之间保持连续。

当你想要一个刻意风格化的呈现时使用 MusicNet，并在输出上标明这一点。

## 返回结果

`sonify()` 返回 `SonificationResult`，其中包含音频以及描述本次转换所需的有效运行设置：

- 解析出的数据类型和方法；
- 物理时长、目标时长和实际输出时长；
- 重复次数、播放速度和音高模式；
- 只读的方法参数与后处理参数；
- 各阶段的原生样本数与时间缩放比；
- 采样率、来源名称和输出路径。

五个底层函数同样可用，返回 `(audio_array, sample_rate)`：

```python
rs.profile_to_wave(...)
rs.amplitude_modulate(...)
rs.griffinlim(...)
rs.hifigan(...)
rs.musicnet(...)
```

需要方法原生时长时使用它们；需要 MSP 统一处理物理时长、方法兼容性和公共元数据时，
使用 `sonify()`。

## 科学说明

MSP 提供以下保证：

- 公开数值输入均为实数、有限、非空，并经过维度校验；
- 控制参数和输出路径在任何耗时推理开始前完成校验；
- 重分箱覆盖完整的源坐标轴，并保持其面积均值；
- 输出具有精确样本数、有限样本值、归零的两端和 `0.9` 的峰值，保存为 PCM16 WAV；
- 下载资源固定在同一个 Hugging Face revision 上，模型加载后调用者的随机数状态保持原样。

这些输出是声化结果：为聆听、探索和交流而设计的听觉表示。由这一目的可以推出几条性质：

- `profile` 和 `amplitude` 会把动态谱沿频率轴归纳；
- Griffin–Lim 估计相位，结果可能带有金属感；
- HiFi-GAN 携带语音模型先验，会影响音色；
- Griffin–Lim 保留首尾的低能量帧，使事件停留在观测时间轴上的真实位置；
- 峰值归一化保持文件内部的结构关系，绝对幅度的比较请回到原始数组；
- `preserve_pitch=True` 使用相位声码器，更适合持续性素材而非极短瞬变。

用于科学工作时，请把原始数组、坐标轴定标、切片时长、MSP 版本和生效参数与 WAV 一并保存。

## 项目结构

```text
MSP/
├── src/radiosonify/
│   ├── inputs.py          # 不可变的科学输入快照
│   ├── registry.py        # 方法兼容性与默认值
│   ├── core.py            # 数值校验、重分箱与 WAV 读写
│   ├── timing.py          # 时长、速度与输出整形
│   ├── api.py             # 统一编排与溯源信息
│   ├── profile.py         # 轮廓插值与乐器响应
│   ├── amplitude.py       # 正弦载波振幅映射
│   ├── griffinlim.py      # 迭代式二维幅度重建
│   ├── hifigan.py         # 带缓存的 HiFi-GAN 推理封装
│   ├── musicnet.py        # 带种子的 MusicNet 后处理
│   └── models/            # 与 checkpoint 兼容的内置模型层 + 许可证
├── tests/
├── examples/sonify_example.py
├── assets/
├── pyproject.toml
└── README_CN.md
```

`MSP/` 可以独立安装和运行。把整个目录复制到任何地方，安装后提供自己的数组和输出路径即可。

## 开发

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

CI 在 Python 3.9 到 3.13 上运行同样的门禁，并在安装可选依赖的任务中执行神经后端的
契约测试。在新的 PyTorch 或 CUDA 环境中依赖神经方法之前，建议先在本地跑一次真实
checkpoint 的冒烟测试。

完整流程见
[CONTRIBUTING.md](https://github.com/SukiYume/MSP/blob/main/CONTRIBUTING.md)。

## 引用与许可

用于科学或面向公众的工作时，请记录 MSP 版本、数据时长、播放速度、解析出的方法、
参数、输入维度和模型 revision。

MSP 自有代码采用
[MIT License](https://github.com/SukiYume/MSP/blob/main/LICENSE)。内置的 MusicNet 推理子集及其 checkpoint
采用 CC BY-NC 4.0，只允许非商业用途。因此发行包元数据使用组合表达式
`MIT AND CC-BY-NC-4.0`。重新分发或使用神经资产前，请阅读
[THIRD_PARTY_NOTICES.md](https://github.com/SukiYume/MSP/blob/main/THIRD_PARTY_NOTICES.md)
和 [MODEL_ASSETS.md](https://github.com/SukiYume/MSP/blob/main/MODEL_ASSETS.md)。

---

<p align="center"><sub>MSP · 让射电脉冲结构可听</sub></p>
