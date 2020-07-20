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

def get_coordinates(a, b, p):
    pa = p - a
    ba = b - a
    t = torch.sum(pa * ba) / torch.sum(ba * ba)
    d = torch.norm(pa - t * ba, 2)
    return t, d

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
        "text_only": args.text_only,
    }
    params = conf["params"]
    params["vae_params"]["vocab"] = vocab
    params["vae_params"]["device"] = device
    params["vae_params"]["text_only"] = args.text_only
    params["vae_params"]["mlp_ni"] = dev_feat.shape[1]
    kwargs = dict(kwargs, **params)

    model = DecomposedVAE(**kwargs)
    model.load(args.load_path)
    model.vae.eval()

    train_data, train_feat = train_data.create_data_batch_feats(32, train_feat, device)
    print("Collecting training distributions...")
    mus, logvars = [], []
    step = 0
    for batch_data, batch_feat in zip(train_data, train_feat):
        mu1, logvar1 = model.vae.lstm_encoder(batch_data)
        mu2, logvar2 = model.vae.mlp_encoder(batch_feat)
        r, _ = model.vae.mlp_encoder(batch_feat, True)
        p = model.vae.get_var_prob(r)
        mu = torch.cat([mu1, mu2], -1)
        logvar = torch.cat([logvar1, logvar2], -1)
        mus.append(mu.detach().cpu())
        logvars.append(logvar.detach().cpu())
        step += 1
        if step % 100 == 0:
            torch.cuda.empty_cache()
    mus = torch.cat(mus, 0)
    logvars = torch.cat(logvars, 0)

    if args.text_only:
        neg_sample = dev_data.data[:10]
        neg_inputs, _ = dev_data._to_tensor(neg_sample, batch_first=False, device=device)
    else:
        neg_sample = dev_feat[:10]
        neg_inputs = torch.tensor(
            neg_sample, dtype=torch.float, requires_grad=False, device=device)
    r, _ = model.vae.mlp_encoder(neg_inputs, True)
    p = model.vae.get_var_prob(r).mean(0, keepdim=True)
    neg_idx = torch.max(p, 1)[1].item()

    if args.text_only:
        pos_sample = dev_data.data[-10:]
        pos_inputs, _ = dev_data._to_tensor(pos_sample, batch_first=False, device=device)
    else:
        pos_sample = dev_feat[-10:]
        pos_inputs = torch.tensor(
            pos_sample, dtype=torch.float, requires_grad=False, device=device)
    r, _ = model.vae.mlp_encoder(pos_inputs, True)
    p = model.vae.get_var_prob(r).mean(0, keepdim=True)
    top2 = torch.topk(p, 2, 1)[1].squeeze()
    if top2[0].item() == neg_idx:
        print("Collision!!! Use second most as postive.")
        pos_idx = top2[1].item()
    else:
        pos_idx = top2[0].item()
    other_idx = -1
    for i in range(3):
        if i not in [pos_idx, neg_idx]:
            other_idx = i
            break

    print("Negative: %d" % neg_idx)
    print("Positive: %d" % pos_idx)

    sep_id = -1
    for idx, x in enumerate(test_data.labels):
        if x == 1:
            sep_id = idx
            break

    bsz = 64
    ori_logps = []
    tra_logps = []
    pos_z2 = model.vae.mlp_encoder.var_embedding[pos_idx:pos_idx + 1]
    neg_z2 = model.vae.mlp_encoder.var_embedding[neg_idx:neg_idx + 1]
    other_z2 = model.vae.mlp_encoder.var_embedding[other_idx:other_idx + 1]
    _, d0 = get_coordinates(pos_z2[0], neg_z2[0], other_z2[0])
    ori_obs = []
    tra_obs = []
    with open(os.path.join(args.load_path, 'generated_results.txt'), "w") as f:
        idx = 0
        step = 0
        n_samples = len(test_data.labels)
        while idx < n_samples:
            label = test_data.labels[idx]
            _idx = idx + bsz if label else min(idx + bsz, sep_id)
            _idx = min(_idx, n_samples)
            var_id = neg_idx if label else pos_idx
            text, _ = test_data._to_tensor(
                test_data.data[idx:_idx], batch_first=False, device=device)
            feat = torch.tensor(test_feat[idx:_idx], dtype=torch.float, requires_grad=False, device=device)
            z1, _ = model.vae.lstm_encoder(text[:min(text.shape[0], 10)])
            ori_z2, _ = model.vae.mlp_encoder(feat)
            tra_z2 = model.vae.mlp_encoder.var_embedding[var_id:var_id + 1, :].expand(
                _idx - idx, -1)
            texts = model.vae.decoder.beam_search_decode(z1, tra_z2)
            for text in texts:
                f.write("%d\t%s\n" % (1 - label, " ".join(text[1:-1])))

            ori_z = torch.cat([z1, ori_z2], -1)
            tra_z = torch.cat([z1, tra_z2], -1)
            for i in range(_idx - idx):
                ori_logps.append(cal_log_density(mus, logvars, ori_z[i:i + 1].cpu()))
                tra_logps.append(cal_log_density(mus, logvars, tra_z[i:i + 1].cpu()))

            idx = _idx
            step += 1
            if step % 100 == 0:
                print(step, idx)

    with open(os.path.join(args.load_path, 'nll.txt'), "w") as f:
        for x, y in zip(ori_logps, tra_logps):
            f.write("%f\t%f\n" % (x, y))

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--feat', type=str, default='glove')
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--text_only', default=False, action='store_true')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
