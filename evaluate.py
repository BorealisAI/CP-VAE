# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pandas as pd
import argparse
from utils.bleu import compute_bleu
from utils.text_utils import MonoTextData
import torch
from classifier import CNNClassifier, evaluate
import os

def main(args):
    data_pth = "data/%s" % args.data_name
    train_pth = os.path.join(data_pth, "train_data.txt")
    train_data = MonoTextData(train_pth, True, vocab=100000)
    vocab = train_data.vocab
    source_pth = os.path.join(data_pth, "test_data.txt")
    target_pth = args.target_path
    eval_data = MonoTextData(target_pth, True, vocab=vocab)
    source = pd.read_csv(source_pth, names=['label', 'content'], sep='\t')
    target = pd.read_csv(target_pth, names=['label', 'content'], sep='\t')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Classification Accuracy
    model = CNNClassifier(len(vocab), 300, [1, 2, 3, 4, 5], 500, 0.5).to(device)
    model.load_state_dict(torch.load("checkpoint/%s-classifier.pt" % args.data_name))
    model.eval()
    eval_data, eval_label = eval_data.create_data_batch_labels(64, device, batch_first=True)
    acc = 100 * evaluate(model, eval_data, eval_label)
    print("Acc: %.2f" % acc)

    # BLEU Score
    total_bleu = 0.0
    sources = []
    targets = []
    for i in range(source.shape[0]):
        s = source.content[i].split()
        t = target.content[i].split()
        sources.append([s])
        targets.append(t)

    total_bleu += compute_bleu(sources, targets)[0]
    total_bleu *= 100
    print("Bleu: %.2f" % total_bleu)

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--target_path', type=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
