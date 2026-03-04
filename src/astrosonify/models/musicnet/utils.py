# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time

import numpy


class timeit:
    def __init__(self, name, logger=None):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger is None:
            print(f'{self.name} took {(time.time() - self.start) * 1000} ms')
        else:
            self.logger.debug('%s took %s ms', self.name, (time.time() - self.start) * 1000)


def mu_law(x, mu=255):
    x = numpy.clip(x, -1, 1)
    x_mu = numpy.sign(x) * numpy.log(1 + mu*numpy.abs(x))/numpy.log(1 + mu)
    return ((x_mu + 1)/2 * mu).astype('int16')


def inv_mu_law(x, mu=255.0):
    x = numpy.array(x).astype(numpy.float32)
    y = 2. * (x - (mu+1.)/2.) / (mu+1.)
    return numpy.sign(y) * (1./mu) * ((1. + mu)**numpy.abs(y) - 1.)
