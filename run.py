# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from utils.exp_utils import create_exp_dir
from utils.text_utils import MonoTextData
import argparse
import os
import torch
import time
import config
from models.decomposed_vae import DecomposedVAE
import numpy as np

def main(args):
    conf = config.CONFIG[args.data_name]
    data_pth = "data/%s" % args.data_name
    train_data_pth = os.path.join(data_pth, "train_data.txt")
    train_feat_pth = os.path.join(data_pth, "train_%s.npy" % args.feat)
    train_data = MonoTextData(train_data_pth, True)
    train_feat = np.load(train_feat_pth)

    vocab = train_data.vocab
    print('Vocabulary size: %d' % len(vocab))

    dev_data_pth = os.path.join(data_pth, "dev_data.txt")
    dev_feat_pth = os.path.join(data_pth, "dev_%s.npy" % args.feat)
    dev_data = MonoTextData(dev_data_pth, True, vocab=vocab)
    dev_feat = np.load(dev_feat_pth)
    test_data_pth = os.path.join(data_pth, "test_data.txt")
    test_feat_pth = os.path.join(data_pth, "test_%s.npy" % args.feat)
    test_data = MonoTextData(test_data_pth, True, vocab=vocab)
    test_feat = np.load(test_feat_pth)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = '{}-{}-{}'.format(args.save, args.data_name, args.feat)
    save_path = os.path.join(save_path, time.strftime("%Y%m%d-%H%M%S"))
    scripts_to_save = [
        'run.py', 'models/decomposed_vae.py', 'models/vae.py',
        'models/base_network.py', 'config.py']
    logging = create_exp_dir(save_path, scripts_to_save=scripts_to_save,
                             debug=args.debug)

    if args.text_only:
        train = train_data.create_data_batch(args.bsz, device)
        dev = dev_data.create_data_batch(args.bsz, device)
        test = test_data.create_data_batch(args.bsz, device)
        feat = train
    else:
        train = train_data.create_data_batch_feats(args.bsz, train_feat, device)
        dev = dev_data.create_data_batch_feats(args.bsz, dev_feat, device)
        test = test_data.create_data_batch_feats(args.bsz, test_feat, device)
        feat = train_feat

    kwargs = {
        "train": train,
        "valid": dev,
        "test": test,
        "feat": feat,
        "bsz": args.bsz,
        "save_path": save_path,
        "logging": logging,
        "text_only": args.text_only,
    }
    params = conf["params"]
    params["vae_params"]["vocab"] = vocab
    params["vae_params"]["device"] = device
    params["vae_params"]["text_only"] = args.text_only
    params["vae_params"]["mlp_ni"] = train_feat.shape[1]
    kwargs = dict(kwargs, **params)

    model = DecomposedVAE(**kwargs)
    try:
        valid_loss = model.fit()
        logging("val loss : {}".format(valid_loss))
    except KeyboardInterrupt:
        logging("Exiting from training early")

    model.load(save_path)
    test_loss = model.evaluate(model.test_data, model.test_feat)
    logging("test loss: {}".format(test_loss[0]))
    logging("test recon: {}".format(test_loss[1]))
    logging("test kl1: {}".format(test_loss[2]))
    logging("test kl2: {}".format(test_loss[3]))
    logging("test mi1: {}".format(test_loss[4]))
    logging("test mi2: {}".format(test_loss[5]))

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp',
                        help='data name')
    parser.add_argument('--save', type=str, default='checkpoint/ours',
                        help='directory name to save')
    parser.add_argument('--bsz', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--text_only', default=False, action='store_true',
                        help='use text only without feats')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='enable debug mode')
    parser.add_argument('--feat', type=str, default='glove',
                        help='feat repr')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
