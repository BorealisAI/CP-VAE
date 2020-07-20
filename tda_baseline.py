# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import baseline_config as config
import torch
from utils.text_utils import MonoTextData
from models.aggressive_vae import AgressiveVAE
import argparse
import numpy as np
import os
from utils.dist_utils import cal_log_density
import kmapper as km
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(args):
    conf = config.CONFIG[args.data_name]
    data_pth = "data/%s" % args.data_name
    train_data_pth = os.path.join(data_pth, "train_data.txt")
    train_data = MonoTextData(train_data_pth, True)
    vocab = train_data.vocab
    dev_data_pth = os.path.join(data_pth, "dev_data.txt")
    dev_data = MonoTextData(dev_data_pth, True, vocab=vocab)
    test_data_pth = os.path.join(data_pth, "test_data.txt")
    test_data = MonoTextData(test_data_pth, True, vocab=vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {
        "train": [1],
        "valid": None,
        "test": None,
        "bsz": 32,
        "save_path": args.load_path,
        "logging": None,
    }
    params = conf["params"]
    params["vae_params"]["vocab"] = vocab
    params["vae_params"]["device"] = device
    kwargs = dict(kwargs, **params)

    model = AgressiveVAE(**kwargs)
    model.load(args.load_path)
    model.vae.eval()

    bsz = 64
    zs = []
    idx = 0
    step = 0
    n_samples = len(train_data.labels)
    n = 10000
    selected_index = np.random.permutation(np.arange(n_samples))[:n]
    while idx < n:
        label = train_data.labels[idx]
        _idx = idx + bsz
        _idx = min(_idx, n)
        inputs = []
        for i in range(idx, _idx):
            inputs.append(train_data.data[selected_index[i]])
        text, _ = train_data._to_tensor(inputs, batch_first=False, device=device)
        z, _ = model.vae.encode(text, 10)
        z = z.squeeze().cpu().detach().numpy()
        zs.append(z[:, :, :16].reshape(-1, 16))

        idx = _idx
        step += 1
        if step % 100 == 0:
            print(step, idx)

    zs = np.vstack(zs)
    mapper = km.KeplerMapper(verbose=1)
    z_embed = mapper.fit_transform(zs, projection='sum')
    graph = mapper.map(z_embed, zs,
            clusterer=sklearn.cluster.DBSCAN(eps=0.1, min_samples=3, metric='cosine'),
            cover=km.Cover(n_cubes=args.resolution, perc_overlap=0.4))
    mapper.visualize(graph, path_html='plot/tda_baseline.html', title='tda baseline')

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--resolution', type=int, default=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
