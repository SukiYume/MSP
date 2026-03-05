# src/radiosonify/models/hifigan/env.py
"""HiFi-GAN environment utilities.

Adapted from https://github.com/jik876/hifi-gan (MIT License).
"""


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
