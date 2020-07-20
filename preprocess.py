# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
import argparse
import os
import pandas as pd
import config

def preprocess(split='train', unk='_UNK'):
    input_file = "data/yelp_data/yelp.{}.txt".format(split)
    output_file = "data/yelp_data/_{}.txt".format(split)
    with open(input_file) as f_in:
        f_out = open(output_file, "w")
        for content in f_in.readlines():
            content = content.split('\t')[1]
            content = content.replace(unk, UNK_TOKEN)
            content = content.replace('-lrb-', '(')
            content = content.replace('-rrb-', ')')
            content = content.replace('-lsb-', '[')
            content = content.replace('-rsb-', ']')
            content = content.replace('-lcb-', '{')
            content = content.replace('-rcb-', "}")
            f_out.write(content)

def get_glove_embeds(in_file, out_file):
    glove_file = "data/glove.840B.300d.txt"
    word_vec = {}
    with open(glove_file) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            word_vec[word] = np.fromstring(vec, sep=' ')

    embeds = []
    with open(in_file) as f:
        for line in f:
            tokens = line.strip().split()
            vec = np.zeros(300, dtype=np.float32)
            sent_len = 0
            for token in tokens:
                if token in word_vec:
                    vec += word_vec[token]
                    sent_len += 1
            if sent_len > 0:
                vec = np.true_divide(vec, sent_len)
            vec = vec.reshape(1, 300)
            embeds.append(vec)
        embeds = np.concatenate(embeds)
        np.save(out_file, embeds)


def concat_files(pth0, pth1, outpth, with_label=True):
    with open(outpth, "w") as f_out:
        with open(pth0, errors='ignore') as f0:
            for line in f0.readlines():
                if with_label:
                    f_out.write("0\t")
                f_out.write(line.strip() + "\n")
        with open(pth1, errors='ignore') as f1:
            for line in f1.readlines():
                if with_label:
                    f_out.write("1\t")
                f_out.write(line.strip() + "\n")

def flip_files(pth, outpth, with_label=True):
    with open(outpth, "w") as f_out:
        with open(pth, errors='ignore') as f:
            for line in f.readlines():
                label, content = line.strip().split("\t")
                if with_label:
                    f_out.write("%d\t" % (1 - int(label)))
                f_out.write(content + "\n")

def main(args):
    data_pth = "data/%s" % args.data_name
    res_pth = "results/%s" % args.data_name
    for split in ["train", "dev", "test"]:
        pth0 = "sentiment.%s.0" % split
        pth1 = "sentiment.%s.1" % split
        outpth = "%s_data.txt" % split
        _outpth = "_%s_data.txt" % split
        pth0 = os.path.join(data_pth, pth0)
        pth1 = os.path.join(data_pth, pth1)
        outpth = os.path.join(data_pth, outpth)
        _outpth = os.path.join(data_pth, _outpth)
        concat_files(pth0, pth1, outpth)
        concat_files(pth0, pth1, _outpth, False)

        fin = _outpth
        fout = os.path.join(data_pth, "%s_glove.npy" % split)
        get_glove_embeds(fin, fout)

    conf = config.CONFIG[args.data_name]
    if "ref0" in conf:
        ref_pth0 = conf["ref0"]
        ref_pth1 = conf["ref1"]
        ref_pth0 = os.path.join(res_pth, ref_pth0)
        ref_pth1 = os.path.join(res_pth, ref_pth1)
        df0 = pd.read_csv(ref_pth0)
        df1 = pd.read_csv(ref_pth1)

        mappings = {
            "CROSSALIGNED": "CA",
            "STYLEEMBEDDING": "SE",
            "MULTIDECODER": "MD",
            "DELETEONLY": "D",
            "DELETEANDRETRIEVE": "D&R",
            "BERT_RET_TFIDF": "G-GST",
            "BERT_DEL": "B-GST",
            "HUMAN": "HUMAN",
            "Source": "SOURCE"
        }

        for model in ["CROSSALIGNED", "STYLEEMBEDDING", "MULTIDECODER", "DELETEONLY",
                      "DELETEANDRETRIEVE", "BERT_RET_TFIDF", "BERT_DEL", "HUMAN", "Source"]:
            name = mappings[model]
            sent0 = df0[model].tolist()
            sent1 = df1[model].tolist()
            outpth = os.path.join(res_pth, name + ".txt")
            with open(outpth, "w") as f_out:
                for x in sent0:
                    f_out.write("1\t%s\n" % str(x).strip().lower())
                for x in sent1:
                    f_out.write("0\t%s\n" % str(x).strip().lower())


def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
