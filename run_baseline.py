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
import baseline_config as config
from models.aggressive_vae import AgressiveVAE

def main(args):
    conf = config.CONFIG[args.data_name]
    data_pth = "data/%s" % args.data_name
    train_data_pth = os.path.join(data_pth, "train_data.txt")
    train_data = MonoTextData(train_data_pth, True)

    vocab = train_data.vocab
    print('Vocabulary size: %d' % len(vocab))

    dev_data_pth = os.path.join(data_pth, "dev_data.txt")
    dev_data = MonoTextData(dev_data_pth, True, vocab=vocab)
    test_data_pth = os.path.join(data_pth, "test_data.txt")
    test_data = MonoTextData(test_data_pth, True, vocab=vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = '{}-{}'.format(args.save, args.data_name)
    save_path = os.path.join(save_path, time.strftime("%Y%m%d-%H%M%S"))
    scripts_to_save = [
        'run.py', 'models/aggressive_vae.py', 'models/vae.py',
        'models/base_network.py', 'config.py']
    logging = create_exp_dir(save_path, scripts_to_save=scripts_to_save,
                             debug=args.debug)

    train = train_data.create_data_batch(args.bsz, device)
    dev = dev_data.create_data_batch(args.bsz, device)
    test = test_data.create_data_batch(args.bsz, device)

    kwargs = {
        "train": train,
        "valid": dev,
        "test": test,
        "bsz": args.bsz,
        "save_path": save_path,
        "logging": logging,
    }
    params = conf["params"]
    params["vae_params"]["vocab"] = vocab
    params["vae_params"]["device"] = device
    kwargs = dict(kwargs, **params)

    model = AgressiveVAE(**kwargs)
    try:
        valid_loss = model.fit()
        logging("val loss : {}".format(valid_loss))
    except KeyboardInterrupt:
        logging("Exiting from training early")

    model.load(save_path)
    test_loss = model.evaluate(model.test_data)
    logging("test loss: {}".format(test_loss[0]))
    logging("test recon: {}".format(test_loss[1]))
    logging("test kl: {}".format(test_loss[2]))
    logging("test mi: {}".format(test_loss[3]))

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp',
                        help='data name')
    parser.add_argument('--save', type=str, default='checkpoint/baseline',
                        help='directory name to save')
    parser.add_argument('--bsz', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='enable debug mode')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
