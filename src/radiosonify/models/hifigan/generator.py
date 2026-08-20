# src/radiosonify/models/hifigan/generator.py
"""HiFi-GAN Generator for mel-spectrogram to waveform conversion.

Adapted from https://github.com/jik876/hifi-gan (MIT License).
"""

from __future__ import annotations

import importlib
from types import ModuleType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d

_parametrizations: ModuleType | None
_parametrize: ModuleType | None
try:
    # PyTorch >= 2.0: use the non-deprecated parametrizations API.
    _parametrizations = importlib.import_module("torch.nn.utils.parametrizations")
    _parametrize = importlib.import_module("torch.nn.utils.parametrize")
except ImportError:
    _parametrizations = None
    _parametrize = None


def _apply_weight_norm(module: nn.Module) -> nn.Module:
    if _parametrizations is not None:
        return _parametrizations.weight_norm(module)
    return torch.nn.utils.weight_norm(module)


def _remove_weight_norm(module: nn.Module) -> nn.Module:
    if _parametrize is not None:
        return _parametrize.remove_parametrizations(module, "weight")
    return torch.nn.utils.remove_weight_norm(module)

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(torch.nn.Module):
    def __init__(self, _h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            _apply_weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            _apply_weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            _apply_weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs2 = nn.ModuleList([
            _apply_weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            _apply_weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            _apply_weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for conv in self.convs1:
            _remove_weight_norm(conv)
        for conv in self.convs2:
            _remove_weight_norm(conv)


class ResBlock2(torch.nn.Module):
    def __init__(self, _h, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            _apply_weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            _apply_weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for conv in self.convs:
            _remove_weight_norm(conv)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        final_channels = h.upsample_initial_channel // (2 ** self.num_upsamples)
        self.conv_pre = _apply_weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(_apply_weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = _apply_weight_norm(Conv1d(final_channels, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for layer in self.ups:
            _remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        _remove_weight_norm(self.conv_pre)
        _remove_weight_norm(self.conv_post)
