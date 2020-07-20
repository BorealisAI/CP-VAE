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

    train = train_data.create_data_batch(32, device)
    dev, dev_labels = dev_data.create_data_batch_labels(64, device)
    dev_labels = [x for sublist in dev_labels for x in sublist]

    print("Collecting training distributions...")
    mus, logvars = [], []
    step = 0
    for batch_data in train:
        mu, logvar = model.vae.encoder(batch_data)
        mus.append(mu.detach().cpu())
        logvars.append(logvar.detach().cpu())
        step += 1
        if step % 100 == 0:
            torch.cuda.empty_cache()
    mus = torch.cat(mus, 0)
    logvars = torch.cat(logvars, 0)

    zs = []
    for batch_data in dev:
        z, _ = model.vae.encoder(batch_data)
        zs.append(z)

    zs = torch.cat(zs, 0)
    mu = zs.mean(dim=0, keepdim=True)
    # unnormalized_zs = zs.data.cpu().numpy()
    zs = (zs - mu).data.cpu().numpy()

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    best_acc = 0.0
    best_idx = -1
    other_idx = 64
    sign = 1
    for i in range(zs.shape[1]):
        correct_num = 0
        for j in range(zs.shape[0]):
            logit = sigmoid(-zs[j, i])
            if np.abs(dev_labels[j] - logit) < 0.5:
                correct_num += 1
        acc = correct_num / zs.shape[0]
        if acc > best_acc:
            best_acc = acc
            best_idx = i
            sign = 1
        if 1 - acc > best_acc:
            best_acc = 1 - acc
            best_idx = i
            sign = 0
    print(best_acc, best_idx)

    v = mus[:, best_idx]
    mu = v.mean()
    std = v.std()
    if args.type == 3:
        max_v = max(v)
        min_v = min(v)
    else:
        max_v = mu + args.type * std
        min_v = mu - args.type * std

    sep_id = -1
    for idx, x in enumerate(test_data.labels):
        if x == 1:
            sep_id = idx
            break

    bsz = 64
    ori_logps = []
    tra_logps = []
    with open(os.path.join(args.load_path, 'generated_text_%d.txt' % args.type), "w") as f:
        idx = 0
        step = 0
        n_samples = len(test_data.labels)
        while idx < n_samples:
            label = test_data.labels[idx]
            _idx = idx + bsz if label else min(idx + bsz, sep_id)
            _idx = min(_idx, n_samples)
            text, _ = test_data._to_tensor(test_data.data[idx:_idx], batch_first=False, device=device)
            z, _ = model.vae.encoder(text)
            ori_z = z.clone()
            tmp = max_v if label == sign else min_v
            if args.type > 0:
                z[:, best_idx] += torch.ones(text.shape[1]).to(device) * tmp
            texts = model.vae.decoder.beam_search_decode(z)
            for text in texts:
                f.write("%d\t%s\n" % (1 - label, " ".join(text[1:-1])))

            for i in range(_idx - idx):
                ori_logps.append(cal_log_density(mus, logvars, ori_z[i:i + 1].cpu()))
                tra_logps.append(cal_log_density(mus, logvars, z[i:i + 1].cpu()))

            idx = _idx
            step += 1
            if step % 100 == 0:
                print(step, idx)

    with open(os.path.join(args.load_path, "nll_%d.txt" % args.type), "w") as f:
        for x, y in zip(ori_logps, tra_logps):
            f.write("%f\t%f\n" % (x, y))

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--type', type=int, default=0,
                        help='0: no change, 1: one std, 2: two std, 3: extreme')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
