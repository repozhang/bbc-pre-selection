import torch
import numpy as np
import random
import time
import codecs
from Constants import *
from torch.distributions.categorical import *
import torch.nn.functional as F
from modules.Utils import *

def get_ms():
    return time.time() * 1000


def init_seed(seed=None):
    if seed is None:
        seed = int(get_ms() // 1000)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def importance_sampling(prob,topk):
    m = Categorical(logits=prob)
    indices = m.sample((topk,)).transpose(0,1)  # batch, topk

    values = prob.gather(1, indices)
    return values, indices

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    mask= (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    if torch.cuda.is_available():
        mask=mask.cuda()
    return mask

def start_end_mask(starts, ends, max_len):
    batch_size=len(starts)
    mask = torch.arange(1, max_len + 1)
    if torch.cuda.is_available():
        mask = mask.cuda()
    mask = mask.unsqueeze(0).expand(batch_size, -1)
    mask1 = mask >= starts.unsqueeze(1).expand_as(mask)
    mask2 = mask <= ends.unsqueeze(1).expand_as(mask)
    mask = (mask1 * mask2)
    return mask


def decode_to_end(model, data, max_target_length=None, schedule_rate=1, softmax=False):
    tgt = data['output']
    batch_size = tgt.size(0)
    if max_target_length is None:
        max_target_length = tgt.size(1)

    encode_outputs = model.encode(data)
    init_decoder_states = model.init_decoder_states(data, encode_outputs)

    decoder_input = new_tensor([BOS] * batch_size, requires_grad=False)

    prob = torch.ones((batch_size,)) * schedule_rate
    if torch.cuda.is_available():
        prob=prob.cuda()

    all_gen_outputs = list()
    all_decode_outputs = list()
    decoder_states = init_decoder_states

    for t in range(max_target_length):
        # decoder_outputs, decoder_states,...
        decode_outputs = model.decode(
            data, decoder_input, decoder_states, encode_outputs
        )
        # decoder_outputs=decode_outputs[0]
        decoder_states = decode_outputs[1] #[sel_d ecode_output[0],gen_decode_output[0]], gen_decode_output[1]

        output = model.generate(data, decode_outputs, softmax=softmax)

        all_gen_outputs.append(output.unsqueeze(0))
        all_decode_outputs.append(decode_outputs)

        if schedule_rate >=1:
            decoder_input = tgt[:, t]
        elif schedule_rate<=0:
            probs, ids = model.to_word(data, output, 1)
            decoder_input = model.generation_to_decoder_input(data, ids[:, 0])
        else:
            probs, ids = model.to_word(data, output, 1)
            indices = model.generation_to_decoder_input(data, ids[:, 0])

            draws = torch.bernoulli(prob).long()
            decoder_input = tgt[:, t] * draws + indices * (1 - draws)

    all_gen_outputs = torch.cat(all_gen_outputs, dim=0).transpose(0, 1).contiguous()

    return encode_outputs, init_decoder_states, all_decode_outputs, all_gen_outputs

def randomk(gen_output, k=5):
    gen_output[:, PAD] = -float('inf')
    gen_output[:, BOS] = -float('inf')
    gen_output[:, UNK] = -float('inf')
    values, indices = importance_sampling(gen_output, k)
    # words=[[tgt_id2vocab[id.item()] for id in one] for one in indices]
    return values, indices

def topk(gen_output, k=5):
    gen_output[:, PAD] = 0
    gen_output[:, BOS] = 0
    gen_output[:, UNK] = 0
    if k>1:
        values, indices = torch.topk(gen_output, k, dim=1, largest=True,
                                     sorted=True, out=None)
    else:
        values, indices = torch.max(gen_output, dim=1, keepdim=True)
    return values, indices


def copy_topk(gen_output, vocab_map, vocab_overlap, k=5):
    vocab=gen_output[:, :vocab_map.size(-1)]
    dy_vocab=gen_output[:, vocab_map.size(-1):]

    vocab=vocab+torch.bmm(dy_vocab.unsqueeze(1), vocab_map).squeeze(1)
    dy_vocab=dy_vocab*vocab_overlap

    gen_output=torch.cat([vocab, dy_vocab], dim=-1)
    return topk(gen_output, k)

# def copy_topk(gen_output,tgt_id2vocab,dyn_id2vocabs, topk=5):
#     values, indices = torch.topk(gen_output[:, :len(tgt_id2vocab)], topk + 3, dim=1, largest=True,
#                                  sorted=True, out=None)
#
#     copy_gen_output = gen_output[:, len(tgt_id2vocab):]
#
#     k_values = []
#     k_indices = []
#     words = []
#     for b in range(indices.size(0)):
#         temp = dict()
#         for i in range(indices.size(1)):
#             id = indices[b, i].item()
#             if id == PAD or id == UNK or id == BOS:
#                 continue
#             w = tgt_id2vocab[id]
#             temp[w] = (id, values[b, i].item())
#             if len(temp) == topk:
#                 break
#
#         for i in range(copy_gen_output.size(1)):
#             if i >= len(dyn_id2vocabs[b]):
#                 continue
#             w = dyn_id2vocabs[b][i]
#             if w == PAD_WORD or w == BOS_WORD:
#                 continue
#
#             if w not in temp:
#                 # temp[w] = (tgt_vocab2id.get(w, UNK), copy_gen_output[b, i].item())
#                 temp[w] = (i + len(tgt_id2vocab), copy_gen_output[b, i].item())
#             else:
#                 temp[w] = (temp[w][0], temp[w][1] + copy_gen_output[b, i].item())
#
#         k_items = sorted(temp.items(), key=lambda d: d[1][1], reverse=True)[:topk]
#         words.append([i[0] for i in k_items])
#         k_indices.append(new_tensor([[i[1][0] for i in k_items]]))
#         k_values.append(new_tensor([[i[1][1] for i in k_items]]))
#     indices = torch.cat(k_indices, dim=0)
#     values = torch.cat(k_values, dim=0)
#
#     return values, indices

# def copy_topk(gen_output,tgt_id2vocab,tgt_vocab2id, dyn_id2vocabs, topk=5):
#     with torch.no_grad():
#         values, indices = torch.topk(gen_output[:, :len(tgt_id2vocab)], topk+3, dim=1, largest=True,
#                                      sorted=True, out=None)
#
#         copy_gen_output = gen_output[:, len(tgt_id2vocab):]
#
#         k_values = []
#         k_indices = []
#         words = []
#         for b in range(indices.size(0)):
#             temp = dict()
#             for i in range(indices.size(1)):
#                 id = indices[b, i].item()
#                 if id==PAD or id==UNK or id==BOS:
#                     continue
#                 w = tgt_id2vocab[id]
#                 temp[w] = (id, values[b, i].item())
#                 if len(temp)==topk:
#                     break
#
#             for i in range(copy_gen_output.size(1)):
#                 if i>=len(dyn_id2vocabs[b]):
#                     continue
#                 w = dyn_id2vocabs[b][i]
#                 if w == PAD_WORD or w==BOS_WORD:
#                     continue
#
#                 if w not in temp:
#                     # temp[w] = (tgt_vocab2id.get(w, UNK), copy_gen_output[b, i].item())
#                     temp[w] = (i+len(tgt_id2vocab), copy_gen_output[b, i].item())
#                 else:
#                     temp[w] = (temp[w][0], temp[w][1] + copy_gen_output[b, i].item())
#
#             k_items = sorted(temp.items(), key=lambda d: d[1][1],reverse=True)[:topk]
#             words.append([i[0] for i in k_items])
#             k_indices.append(torch.tensor([[i[1][0] for i in k_items]]))
#             k_values.append(torch.tensor([[i[1][1] for i in k_items]]))
#         indices = torch.cat(k_indices, dim=0)
#         values = torch.cat(k_values, dim=0)
#
#     return words, values, indices

def to_sentence(batch_indices, id2vocab):
    batch_size=len(batch_indices)
    summ=list()
    for i in range(batch_size):
        indexes=batch_indices[i]
        text_summ2 = []
        for index in indexes:
            index = index.item()
            w = id2vocab[index]
            if w == BOS_WORD or w == PAD_WORD:
                continue
            if w == EOS_WORD:
                break
            text_summ2.append(w)
        if len(text_summ2)==0:
            text_summ2.append(UNK_WORD)
        summ.append(text_summ2)
    return summ

def to_copy_sentence(data, batch_indices,tgt_id2vocab, dyn_id2vocab_map):
    ids=data['id']
    batch_size=len(batch_indices)
    summ=list()
    for i in range(batch_size):
        indexes=batch_indices[i]
        text_summ2 = []
        dyn_id2vocab=dyn_id2vocab_map[ids[i].item()]
        for index in indexes:
            index = index.item()
            if index < len(tgt_id2vocab):
                w = tgt_id2vocab[index]
            elif index - len(tgt_id2vocab) in dyn_id2vocab:
                w = dyn_id2vocab[index - len(tgt_id2vocab)]
            else:
                w = PAD_WORD

            if w == BOS_WORD or w == PAD_WORD:
                continue

            if w == EOS_WORD:
                break

            text_summ2.append(w)

        if len(text_summ2)==0:
            text_summ2.append(UNK_WORD)

        summ.append(text_summ2)
    return summ

def position(seq):
    pos=new_tensor([i+1 for i in range(seq.size(1))],requires_grad=False).long()
    pos=pos.repeat(seq.size(0),1)
    pos=pos.mul(seq.ne(PAD).long()).long()
    return pos