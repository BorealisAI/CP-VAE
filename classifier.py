# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from utils.text_utils import MonoTextData
import numpy as np
import os

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, filter_sizes, n_filters, dropout):
        super(CNNClassifier, self).__init__()
        self.n_filters = n_filters

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cnns = nn.ModuleList([
            nn.Conv2d(embed_dim, n_filters, (x, 1)) for x in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(len(filter_sizes) * n_filters, 1)

    def forward(self, inputs):
        inputs = self.embedding(inputs).unsqueeze(-1)
        inputs = inputs.permute(0, 2, 1, 3)
        outputs = []
        for cnn in self.cnns:
            conv = cnn(inputs)
            h = F.leaky_relu(conv)
            pooled = torch.max(h, 2)[0].view(-1, self.n_filters)
            outputs.append(pooled)
        outputs = torch.cat(outputs, -1)
        outputs = self.dropout(outputs)
        logits = self.output(outputs)
        return logits.squeeze(1)

def evaluate(model, eval_data, eval_label):
    correct_num = 0
    total_sample = 0
    for batch_data, batch_label in zip(eval_data, eval_label):
        batch_size = batch_data.size(0)
        logits = model(batch_data)
        probs = torch.sigmoid(logits)
        y_hat = list((probs > 0.5).long().cpu().numpy())
        correct_num += sum([p == q for p, q in zip(batch_label, y_hat)])
        total_sample += batch_size
    return correct_num / total_sample

def main(args):
    data_pth = "data/%s" % args.data_name
    train_pth = os.path.join(data_pth, "train_data.txt")
    dev_pth = os.path.join(data_pth, "dev_data.txt")
    test_pth = os.path.join(data_pth, "test_data.txt")

    train_data = MonoTextData(train_pth, True, vocab=100000)
    vocab = train_data.vocab
    dev_data = MonoTextData(dev_pth, True, vocab=vocab)
    test_data = MonoTextData(test_pth, True, vocab=vocab)
    path = "checkpoint/%s-classifier.pt" % args.data_name

    glove_embed = np.zeros((len(vocab), 300))
    with open("data/glove.840B.300d.txt") as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in vocab:
                wid = vocab[word]
                glove_embed[wid, :] = np.fromstring(vec, sep=' ', dtype=np.float32)

        _mu = glove_embed.mean()
        _std = glove_embed.std()
        glove_embed[:4, :] = np.random.randn(4, 300) * _std + _mu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_batch, train_label = train_data.create_data_batch_labels(64, device, batch_first=True)
    dev_batch, dev_label = dev_data.create_data_batch_labels(64, device, batch_first=True)
    test_batch, test_label = test_data.create_data_batch_labels(64, device, batch_first=True)

    model = CNNClassifier(len(vocab), 300, [1, 2, 3, 4, 5], 500, 0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    nbatch = len(train_batch)
    best_acc = 0.0
    step = 0

    with torch.no_grad():
        model.embedding.weight.fill_(0.)
        model.embedding.weight += torch.FloatTensor(glove_embed).to(device)

    for epoch in range(args.max_epochs):
        for idx in np.random.permutation(range(nbatch)):
            batch_data = train_batch[idx]
            batch_label = train_label[idx]
            batch_label = torch.tensor(batch_label, dtype=torch.float,
                                       requires_grad=False, device=device)

            optimizer.zero_grad()
            logits = model(batch_data)
            loss = F.binary_cross_entropy_with_logits(logits, batch_label)
            loss.backward()
            optimizer.step()

            step += 1
            if step % 1000 == 0:
                print('Loss: %2f' % loss.item())

        model.eval()
        acc = evaluate(model, dev_batch, dev_label)
        model.train()
        print('Valid Acc: %.2f' % acc)
        if acc > best_acc:
            best_acc = acc
            print('saving to %s' % path)
            torch.save(model.state_dict(), path)

    model.load_state_dict(torch.load(path))
    model.eval()
    acc = evaluate(model, test_batch, test_label)
    print('Test Acc: %.2f' % acc)

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--max_epochs', type=int, default=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
