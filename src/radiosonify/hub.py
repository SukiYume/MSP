"""固定版本的示例数据、模型权重和本地合成乐器响应管理。"""

from __future__ import annotations

import os
import time
import wave
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

REPO_ID = "TorchLight/radiosonify"
# 固定提交可避免上游同名文件变化后，本地结果在不知情时漂移。
REVISION = "14d214896b004b8e38e048b36715362637733114"
CACHE_DIR = os.environ.get(
    "RADIOSONIFY_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "radiosonify"),
)

EXAMPLE_MAP = {
    "burst": "Burst.npy",
    "raw_burst": "RawBurst.npy",
    "parkes_burst": "ParkesBurst.npy",
    "profile": "Profile.npy",
}

INSTRUMENT_MAP = {
    "violin": "violin.wav",
    "piano": "piano.wav",
}

_INSTRUMENT_SAMPLE_RATE = 48_000
_INSTRUMENT_VERSION = "v1"


def _download_with_context(filename: str) -> str:
    """优先命中本地缓存；缓存缺失时最多进行两次联网下载。"""
    try:
        return hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            cache_dir=CACHE_DIR,
            revision=REVISION,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        pass
    except (EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError) as e:
        raise RuntimeError(
            f"Resource '{filename}' not found in Hugging Face repo '{REPO_ID}': {e}"
        ) from e

    last_error = None
    for attempt in range(2):
        try:
            return hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                cache_dir=CACHE_DIR,
                revision=REVISION,
            )
        except LocalEntryNotFoundError as exc:
            # 在线请求也会用该异常报告缓存缺失叠加连接失败。它同时继承
            # EntryNotFoundError，因此必须先捕获，才能保留重试和正确诊断。
            last_error = exc
            if attempt == 0:
                time.sleep(0.3)
        except (EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError) as e:
            raise RuntimeError(
                f"Resource '{filename}' not found in Hugging Face repo '{REPO_ID}': {e}"
            ) from e
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                # 短暂退避只处理瞬时网络错误；资源不存在则在上方立即终止。
                time.sleep(0.3)

    if last_error is None:  # pragma: no cover - range(2) 仅可能通过 return 或异常到达这里
        raise RuntimeError(f"Download of '{filename}' failed without an underlying error")
    raise RuntimeError(
        f"Failed to download '{filename}' from Hugging Face repo '{REPO_ID}'. "
        "Check network connectivity, Hugging Face access permissions, and local cache integrity. "
        f"Original error: {last_error}"
    ) from last_error


def get_data_path(filename: str) -> str:
    """下载示例数据并返回本地缓存路径。"""
    return _download_with_context(f"data/{filename}")


def get_model_path(model_name: str, filename: str) -> str:
    """下载模型文件并返回本地缓存路径。"""
    return _download_with_context(f"models/{model_name}/{filename}")


def _synthesize_instrument(name: str) -> np.ndarray:
    """生成确定性的短乐器脉冲响应，不依赖外部录音素材。"""
    sample_count = int(0.35 * _INSTRUMENT_SAMPLE_RATE)
    t = np.arange(sample_count, dtype=np.float64) / _INSTRUMENT_SAMPLE_RATE

    if name == "violin":
        fundamental = 220.0
        vibrato = 0.003 * np.sin(2 * np.pi * 5.2 * t)
        phase = 2 * np.pi * fundamental * (t + vibrato)
        sound = sum(np.sin(harmonic * phase) / harmonic for harmonic in range(1, 9))
        envelope = (1.0 - np.exp(-t / 0.0025)) * np.exp(-t / 0.24)
    else:
        fundamental = 261.625565
        sound = sum(
            np.cos(2 * np.pi * fundamental * harmonic * t) / harmonic**1.4
            for harmonic in range(1, 8)
        )
        envelope = np.exp(-t / 0.11) * (1.0 - 0.15 * np.exp(-t / 0.004))

    response = sound * envelope
    response -= np.mean(response)
    peak = float(np.max(np.abs(response)))
    if peak == 0:  # pragma: no cover - the analytic signals above are non-zero
        raise RuntimeError(f"failed to synthesize instrument response: {name}")
    return (0.95 * response / peak).astype(np.float32)


def _write_pcm16_atomic(path: Path, audio: np.ndarray) -> None:
    """把生成的单声道响应原子写入缓存，避免并发产生半文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    try:
        with wave.open(str(temporary), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(_INSTRUMENT_SAMPLE_RATE)
            wav.writeframes(pcm.tobytes())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def get_instrument_path(name: str) -> str:
    """生成并缓存 MSP 自有的乐器脉冲响应，返回本地 WAV 路径。"""
    if name not in INSTRUMENT_MAP:
        raise ValueError(f"Unknown instrument: {name}. Available: {list(INSTRUMENT_MAP.keys())}")
    destination = (
        Path(CACHE_DIR) / "generated-instruments" / _INSTRUMENT_VERSION / INSTRUMENT_MAP[name]
    )
    if not destination.is_file():
        _write_pcm16_atomic(destination, _synthesize_instrument(name))
    return str(destination)


def load_example(name: str) -> np.ndarray:
    """按公开名称加载示例数组。

    Args:
        name: One of 'burst', 'raw_burst', 'parkes_burst', 'profile'.
    """
    if name not in EXAMPLE_MAP:
        raise ValueError(f"Unknown example: {name}. Available: {list(EXAMPLE_MAP.keys())}")
    path = get_data_path(EXAMPLE_MAP[name])
    return np.load(path, allow_pickle=False)
