# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright (c) 2020-present, Juxian He
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#################################################################################################
# Code is based on the VAE lagging encoder (https://arxiv.org/abs/1901.05534) implementation
# from https://github.com/jxhe/vae-lagging-encoder by Junxian He
#################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from .utils import log_sum_exp
import numpy as np
from scipy.stats import ortho_group

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class, dropout):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Invalid type for input_dims!'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)

        for i, n_hidden in enumerate(n_hiddens):
            l_i = i + 1
            layers['fc{}'.format(l_i)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(l_i)] = nn.ReLU()
            layers['drop{}'.format(l_i)] = nn.Dropout(dropout)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model = nn.Sequential(layers)

    def forward(self, input):
        return self.model.forward(input)

class GaussianEncoderBase(nn.Module):
    def __init__(self):
        super(GaussianEncoderBase, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def sample(self, inputs, nsamples):
        mu, logvar = self.forward(inputs)
        z = self.reparameterize(mu, logvar, nsamples)
        return z, (mu, logvar)

    def encode(self, inputs, nsamples=1):
        mu, logvar = self.forward(inputs)
        z = self.reparameterize(mu, logvar, nsamples)
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1)
        return z, KL

    def reparameterize(self, mu, logvar, nsamples=1):
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(0).expand(nsamples, batch_size, nz)
        std_expd = std.unsqueeze(0).expand(nsamples, batch_size, nz)

        eps = torch.zeros_like(std_expd).normal_()
        return mu_expd + torch.mul(eps, std_expd)

    def sample_from_inference(self, x, nsamples=1):
        mu, logvar = self.forward(x)
        batch_size, nz = mu.size()
        return mu.unsqueeze(0).expand(nsamples, batch_size, nz)

    def eval_inference_dist(self, x, z, param=None):
        nz = z.size(2)
        if not param:
            mu, logvar = self.forward(x)
        else:
            mu, logvar = param

        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()
        dev = z - mu

        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        return log_density.squeeze(0)

    def calc_mi(self, x):
        mu, logvar = self.forward(x)

        x_batch, nz = mu.size()

        neg_entropy = (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)).mean()

        z_samples = self.reparameterize(mu, logvar, 1)

        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        dev = z_samples - mu

        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        log_qz = log_sum_exp(log_density, dim=0) - math.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()

class LSTMEncoder(GaussianEncoderBase):
    def __init__(self, ni, nh, nz, vocab_size, model_init, emb_init):
        super(LSTMEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, ni)

        self.lstm = nn.LSTM(input_size=ni,
                            hidden_size=nh,
                            num_layers=2,
                            bidirectional=True)
        self.linear = nn.Linear(nh, 2 * nz, bias=False)
        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, inputs):
        if len(inputs.size()) > 2:
            word_embed = torch.matmul(inputs, self.embed.weight)
        else:
            word_embed = self.embed(inputs)

        outputs, (last_state, last_cell) = self.lstm(word_embed)
        seq_len, bsz, hidden_size = outputs.size()
        hidden_repr = outputs.view(seq_len, bsz, 2, -1).mean(2)
        hidden_repr = torch.max(hidden_repr, 0)[0]

        mean, logvar = self.linear(hidden_repr).chunk(2, -1)
        return mean, logvar

class SemLSTMEncoder(GaussianEncoderBase):
    def __init__(self, ni, nh, nz, vocab_size, n_vars, model_init, emb_init, device):
        super(SemLSTMEncoder, self).__init__()
        self.n_vars = n_vars
        self.device = device
        self.embed = nn.Embedding(vocab_size, ni)

        self.lstm = nn.LSTM(input_size=ni,
                            hidden_size=nh,
                            num_layers=1,
                            bidirectional=True)
        self.linear = nn.Linear(nh, 2 * nz, bias=False)
        self.var_embedding = nn.Parameter(torch.zeros((n_vars, nz)))
        self.var_linear = nn.Linear(nz, n_vars)
        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def encode_var(self, inputs, return_p=False):
        logits = self.var_linear(inputs)
        prob = F.softmax(logits, -1)
        if return_p:
            return torch.matmul(prob, self.var_embedding), prob
        return torch.matmul(prob, self.var_embedding)

    def orthogonal_regularizer(self, norm=10):
        tmp = torch.mm(self.var_embedding, self.var_embedding.permute(1, 0))
        return torch.norm(tmp - norm * torch.diag(torch.ones(self.n_vars, device=self.device)), 2)

    def forward(self, inputs, return_origin=False):
        if len(inputs.size()) > 2:
            word_embed = torch.matmul(inputs, self.embed.weight)
        else:
            word_embed = self.embed(inputs)

        outputs, (last_state, last_cell) = self.lstm(word_embed)
        seq_len, bsz, hidden_size = outputs.size()
        hidden_repr = outputs.view(seq_len, bsz, 2, -1).mean(2)
        hidden_repr = torch.max(hidden_repr, 0)[0]

        mean, logvar = self.linear(hidden_repr).chunk(2, -1)
        if return_origin:
            return mean, logvar
        return self.encode_var(mean), logvar

class SemMLPEncoder(GaussianEncoderBase):
    def __init__(self, ni, nz, n_vars, model_init, device):
        super(SemMLPEncoder, self).__init__()
        self.n_vars = n_vars
        self.device = device

        self.output = nn.Linear(ni, 2 * nz)
        self.var_embedding = nn.Parameter(torch.zeros((n_vars, nz)))

        self.var_linear = nn.Linear(nz, n_vars)
        self.reset_parameters(model_init)

    def reset_parameters(self, model_init):
        for param in self.parameters():
            model_init(param)

    def encode_var(self, inputs, return_p=False):
        logits = self.var_linear(inputs)
        prob = F.softmax(logits, -1)
        if return_p:
            return torch.matmul(prob, self.var_embedding), prob
        return torch.matmul(prob, self.var_embedding)

    def orthogonal_regularizer(self, norm=100):
        tmp = torch.mm(self.var_embedding, self.var_embedding.permute(1, 0))
        return torch.norm(tmp - norm * torch.diag(torch.ones(self.n_vars, device=self.device)), 2)

    def forward(self, inputs, return_origin=False):
        mean, logvar = self.output(inputs).chunk(2, -1)
        if return_origin:
            return mean, logvar
        return self.encode_var(mean), logvar

class LSTMDecoder(nn.Module):
    def __init__(self, ni, nh, nz, dropout_in, dropout_out, vocab,
                 model_init, emb_init, device):
        super(LSTMDecoder, self).__init__()
        self.nz = nz
        self.vocab = vocab
        self.device = device

        self.embed = nn.Embedding(len(vocab), ni, padding_idx=-1)

        self.dropout_in = nn.Dropout(dropout_in)
        self.dropout_out = nn.Dropout(dropout_out)

        self.trans_linear = nn.Linear(nz, nh, bias=False)

        self.lstm = nn.LSTM(input_size=ni + nz,
                            hidden_size=nh,
                            num_layers=1)

        self.pred_linear = nn.Linear(nh, len(vocab), bias=False)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, inputs, z):
        n_sample, batch_size, _ = z.size()
        seq_len = inputs.size(0)

        word_embed = self.embed(inputs)
        word_embed = self.dropout_in(word_embed)

        if n_sample == 1:
            z_ = z.expand(seq_len, batch_size, self.nz)
        else:
            raise NotImplementedError

        word_embed = torch.cat((word_embed, z_), -1)

        z = z.view(batch_size * n_sample, self.nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(word_embed, (h_init, c_init))

        output = self.dropout_out(output)
        output_logits = self.pred_linear(output)

        return output_logits.view(-1, batch_size, len(self.vocab))

    def decode(self, z, greedy=True):
        batch_size = z.size(0)
        decoded_batch = [[] for _ in range(batch_size)]

        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor([self.vocab["<s>"]] * batch_size, dtype=torch.long,
                                     device=self.device).unsqueeze(0)
        end_symbol = torch.tensor([self.vocab["</s>"]] * batch_size, dtype=torch.long,
                                  device=self.device)

        mask = torch.ones((batch_size), dtype=torch.uint8, device=self.device)
        length_c = 1
        while mask.sum().item() != 0 and length_c < 100:
            word_embed = self.embed(decoder_input)
            word_embed = torch.cat((word_embed, z.unsqueeze(0)), dim=-1)

            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

            decoder_output = self.pred_linear(output)
            output_logits = decoder_output.squeeze(0)

            if greedy:
                select_index = torch.argmax(output_logits, dim=1)
            else:
                sample_prob = F.softmax(output_logits, dim=1)
                select_index = torch.multinomial(sample_prob, num_samples=1).squeeze(1)

            decoder_input = select_index.unsqueeze(0)
            length_c += 1

            for i in range(batch_size):
                if mask[i].item():
                    decoded_batch[i].append(self.vocab.id2word(select_index[i].item()))

            mask = torch.mul((select_index != end_symbol), mask)

        return decoded_batch

    def beam_search_decode(self, z1, z2=None, K=5, max_t=20):
        decoded_batch = []
        if z2 is not None:
            z = torch.cat([z1, z2], -1)
        else:
            z = z1
        batch_size, nz = z.size()

        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        for idx in range(batch_size):
            decoder_input = torch.tensor([[self.vocab["<s>"]]], dtype=torch.long,
                                         device=self.device)
            decoder_hidden = (h_init[:, idx, :].unsqueeze(1), c_init[:, idx, :].unsqueeze(1))
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0.1, 1)
            live_hypotheses = [node]

            completed_hypotheses = []

            t = 0
            while len(completed_hypotheses) < K and t < max_t:
                t += 1

                decoder_input = torch.cat([node.wordid for node in live_hypotheses], dim=1)

                decoder_hidden_h = torch.cat([node.h[0] for node in live_hypotheses], dim=1)
                decoder_hidden_c = torch.cat([node.h[1] for node in live_hypotheses], dim=1)

                decoder_hidden = (decoder_hidden_h, decoder_hidden_c)

                word_embed = self.embed(decoder_input)
                word_embed = torch.cat((word_embed, z[idx].view(1, 1, -1).expand(
                    1, len(live_hypotheses), nz)), dim=-1)

                output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

                output_logits = self.pred_linear(output)
                decoder_output = F.log_softmax(output_logits, dim=-1)

                prev_logp = torch.tensor([node.logp for node in live_hypotheses],
                                         dtype=torch.float, device=self.device)
                decoder_output = decoder_output + prev_logp.view(1, len(live_hypotheses), 1)

                decoder_output = decoder_output.view(-1)

                log_prob, indexes = torch.topk(decoder_output, K - len(completed_hypotheses))

                live_ids = indexes // len(self.vocab)
                word_ids = indexes % len(self.vocab)

                live_hypotheses_new = []
                for live_id, word_id, log_prob_ in zip(live_ids, word_ids, log_prob):
                    node = BeamSearchNode((
                        decoder_hidden[0][:, live_id, :].unsqueeze(1),
                        decoder_hidden[1][:, live_id, :].unsqueeze(1)),
                        live_hypotheses[live_id], word_id.view(1, 1), log_prob_, t)

                    if word_id.item() == self.vocab["</s>"]:
                        completed_hypotheses.append(node)
                    else:
                        live_hypotheses_new.append(node)

                live_hypotheses = live_hypotheses_new

                if len(completed_hypotheses) == K:
                    break

            for live in live_hypotheses:
                completed_hypotheses.append(live)

            utterances = []
            for n in sorted(completed_hypotheses, key=lambda node: node.logp, reverse=True):
                utterance = []
                utterance.append(self.vocab.id2word(n.wordid.item()))
                while n.prevNode is not None:
                    n = n.prevNode
                    utterance.append(self.vocab.id2word(n.wordid.item()))

                utterance = utterance[::-1]
                utterances.append(utterance)

                break

            decoded_batch.append(utterances[0])

        return decoded_batch
