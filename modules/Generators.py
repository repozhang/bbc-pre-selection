import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from Constants import *

class LinearGenerator(nn.Module):
    def __init__(self, feature_size, tgt_vocab_size, logit_scale=1, weight=None):
        super(LinearGenerator, self).__init__()
        self.linear = nn.Linear(feature_size, tgt_vocab_size)
        self.logit_scale=logit_scale
        if weight is not None:
            self.linear.weight =weight

    def forward(self, feature, softmax=True):
        logits = self.linear(feature) * self.logit_scale
        if softmax:
            logits=F.softmax(logits,dim=1)
        return logits

class CopyGenerator(nn.Module):
    def __init__(self, feature_size, tgt_vocab_size, logit_scale=1, weight=None):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(feature_size, tgt_vocab_size)
        self.linear_copy = nn.Linear(feature_size, 1)
        self.logit_scale = logit_scale
        if weight is not None:
            self.linear.weight = weight

    def forward(self, feature, attention, src_map):
        """
        attention: [batch, source_len] batch can be batch*target_len
        src_map: [batch, source_len, copy_vocab_size] value: 0 or 1
        """
        # CHECKS
        batch, _ = feature.size()
        batch, slen = attention.size()
        batch,slen,cvocab = src_map.size()

        # Original probabilities.
        logits = self.linear(feature) * self.logit_scale
        logits[:, PAD] = -float('inf')
        logits = F.softmax(logits, dim=1)

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(feature))
        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(logits,  1 - p_copy.expand_as(logits))
        # mul_attn = torch.mul(attention, p_copy.expand_as(attention))
        # copy_prob = torch.bmm(mul_attn.view(-1, batch, slen)
        #                       .transpose(0, 1),
        #                       src_map.float()).squeeze(1)

        copy_prob = torch.bmm(attention.view(-1, batch, slen)
                              .transpose(0, 1),
                              src_map.float()).squeeze(1)
        copy_prob = torch.mul(copy_prob, p_copy.expand_as(copy_prob))

        return torch.cat([out_prob, copy_prob], 1)
        # return out_prob, copy_prob

