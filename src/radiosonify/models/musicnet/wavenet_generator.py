# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

import time
import logging

from .wavenet import WaveNet


class QueuedConv1d(nn.Module):
    def __init__(self, conv, data):
        super().__init__()
        if isinstance(conv, nn.Conv1d):
            self.inner_conv = nn.Conv1d(conv.in_channels,
                                        conv.out_channels,
                                        conv.kernel_size)
            self.init_len = conv.padding[0]
            self.inner_conv.weight.data.copy_(conv.weight.data)
            self.inner_conv.bias.data.copy_(conv.bias.data)

        elif isinstance(conv, QueuedConv1d):
            self.inner_conv = nn.Conv1d(conv.inner_conv.in_channels,
                                        conv.inner_conv.out_channels,
                                        conv.inner_conv.kernel_size)
            self.init_len = conv.init_len
            self.inner_conv.weight.data.copy_(conv.inner_conv.weight.data)
            self.inner_conv.bias.data.copy_(conv.inner_conv.bias.data)

        self.init_queue(data)

    def init_queue(self, data):
        self.queue = deque([data[:, :, 0:1]]*self.init_len,
                           maxlen=self.init_len)

    def forward(self, x):
        y = x
        x = torch.cat([self.queue[0], x], dim=2)
        self.queue.append(y)

        return self.inner_conv(x)


class WavenetGenerator(nn.Module):
    Q_ZERO = 128

    def __init__(self, wavenet: WaveNet, batch_size=1, cond_repeat=800, wav_freq=16000):
        super().__init__()
        self.wavenet = wavenet
        self.wavenet.shift_input = False
        self.cond_repeat = cond_repeat
        self.wav_freq = wav_freq
        self.batch_size = batch_size
        self.was_cuda = next(self.wavenet.parameters()).is_cuda

        x = torch.zeros(self.batch_size, 1, 1)
        x = x.cuda() if self.was_cuda else x
        self.wavenet.first_conv = QueuedConv1d(self.wavenet.first_conv, x)

        x = torch.zeros(self.batch_size, self.wavenet.residual_channels, 1)
        x = x.cuda() if self.was_cuda else x
        for layer in self.wavenet.layers:
            layer.causal = QueuedConv1d(layer.causal, x)

        if self.was_cuda:
            self.wavenet.cuda()
        self.wavenet.eval()

    def forward(self, x, c=None):
        return self.wavenet(x, c)

    def reset(self):
        return self.init()

    def init(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size

        x = torch.zeros(self.batch_size, 1, 1)
        x = x.cuda() if self.was_cuda else x
        self.wavenet.first_conv.init_queue(x)

        x = torch.zeros(self.batch_size, self.wavenet.residual_channels, 1)
        x = x.cuda() if self.was_cuda else x
        for layer in self.wavenet.layers:
            layer.causal.init_queue(x)

        if self.was_cuda:
            self.wavenet.cuda()

    @staticmethod
    def softmax_and_sample(prediction, method='sample'):
        if method == 'sample':
            probabilities = F.softmax(prediction, dim=1)
            samples = torch.multinomial(probabilities, 1)
        elif method == 'max':
            _, samples = torch.max(F.softmax(prediction, dim=1), dim=1)
        else:
            assert False, "Method not supported."

        return samples

    def generate(self, encodings, init=True, method='sample', pbar=None):
        if init:
            self.init(encodings.size(0))

        n_enc = encodings.size(2)
        cond_repeat = self.cond_repeat
        total_len = n_enc * cond_repeat + 1

        samples = torch.full(
            (encodings.size(0), 1, total_len),
            self.Q_ZERO, dtype=torch.long,
            device=encodings.device,
        )

        # Pre-bind references to avoid Python attribute lookup in tight loop
        _forward = self.forward
        _softmax = F.softmax
        _multinomial = torch.multinomial
        use_sample = (method == 'sample')

        with torch.inference_mode():
            t0 = time.time()
            for t1 in range(n_enc):
                # Cache conditioning – same for all cond_repeat inner iterations
                c = encodings[:, :, t1:t1 + 1]
                base = t1 * cond_repeat
                for t2 in range(cond_repeat):
                    t = base + t2
                    # No .clone() needed: WaveNet forward creates new tensors
                    prediction = _forward(samples[:, :, t:t + 1], c)[:, :, 0]
                    if use_sample:
                        samples[:, :, t + 1] = _multinomial(
                            _softmax(prediction, dim=1), 1)
                    else:
                        samples[:, :, t + 1] = prediction.argmax(
                            dim=1, keepdim=True)

                if pbar is not None:
                    pbar.update(1)

            logging.info(
                f'{encodings.size(0)} samples of '
                f'{n_enc * cond_repeat / self.wav_freq} seconds length '
                f'generated in {time.time() - t0} seconds.')

        return samples[:, :, 1:]
