# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

CONFIG = {}
CONFIG["yelp"] = {
    "label": True,
    "params": {
        "log_interval": 2000,
        "num_epochs": 100,
        "enc_lr": 1.0,
        "dec_lr": 1.0,
        "warm_up": 10,
        "kl_start": 0.1,
        "aggressive": False,
        "vae_params": {
            "ni": 256,
            "nz": 80,
            "enc_nh": 1024,
            "dec_nh": 1024,
            "dec_dropout_in": 0.5,
            "dec_dropout_out": 0.5,
        }
    }
}
CONFIG["amazon"] = {
    "label": True,
    "params": {
        "log_interval": 2000,
        "num_epochs": 100,
        "enc_lr": 1e-3,
        "dec_lr": 1.0,
        "warm_up": 10,
        "kl_start": 0.1,
        "aggressive": True,
        "vae_params": {
            "ni": 128,
            "nz": 32,
            "enc_nh": 1024,
            "dec_nh": 1024,
            "dec_dropout_in": 0.5,
            "dec_dropout_out": 0.5,
        }
    }
}
