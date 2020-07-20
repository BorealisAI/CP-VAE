# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import config
import torch
from utils.text_utils import MonoTextData
from models.decomposed_vae import DecomposedVAE
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
    train_feat_pth = os.path.join(data_pth, "train_%s.npy" % args.feat)
    train_data = MonoTextData(train_data_pth, True)
    train_feat = np.load(train_feat_pth)
    vocab = train_data.vocab
    dev_data_pth = os.path.join(data_pth, "dev_data.txt")
    dev_feat_pth = os.path.join(data_pth, "dev_%s.npy" % args.feat)
    dev_data = MonoTextData(dev_data_pth, True, vocab=vocab)
    dev_feat = np.load(dev_feat_pth)
    test_data_pth = os.path.join(data_pth, "test_data.txt")
    test_feat_pth = os.path.join(data_pth, "test_%s.npy" % args.feat)
    test_data = MonoTextData(test_data_pth, True, vocab=vocab)
    test_feat = np.load(test_feat_pth)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {
        "train": ([1], None),
        "valid": (None, None),
        "test": (None, None),
        "feat": None,
        "bsz": 32,
        "save_path": args.load_path,
        "logging": None,
        "text_only": False,
    }
    params = conf["params"]
    params["vae_params"]["vocab"] = vocab
    params["vae_params"]["device"] = device
    params["vae_params"]["text_only"] = False
    params["vae_params"]["mlp_ni"] = dev_feat.shape[1]
    kwargs = dict(kwargs, **params)

    model = DecomposedVAE(**kwargs)
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
        text, _ = train_data._to_tensor(
                train_data.data[idx:_idx], batch_first=False, device=device)
        feat = torch.tensor(train_feat[idx:_idx], dtype=torch.float, requires_grad=False, device=device)
        z1, _ = model.vae.lstm_encoder(text)
        z2, _ = model.vae.encode_semantic(feat, 10)
        z = z2.detach().cpu().numpy()
        zs.append(z.reshape(-1, 16))

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
    mapper.visualize(graph, path_html='plot/tda_ours.html', title='tda ours')

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--feat', type=str, default='glove')
    parser.add_argument('--resolution', type=int, default=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
