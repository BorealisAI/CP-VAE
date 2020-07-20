# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import torch
from torch.distributions.normal import Normal

def cal_log_density(mu, logvar, z):
    nz = z.shape[1]
    dev = z.expand_as(mu) - mu
    var = logvar.exp()
    log_density = -0.5 * ((dev ** 2) / var).sum(-1) - \
        0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))
    return log_density.mean().item()
