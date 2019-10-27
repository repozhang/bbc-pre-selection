import torch
import numpy as np
from Constants import *
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence


def new_tensor(array, requires_grad=False):
    tensor=torch.tensor(array, requires_grad=requires_grad)
    if torch.cuda.is_available():
        tensor=tensor.cuda()
    return tensor

# def hotfix_pack_padded_sequence(input, lengths, batch_first=True):
#     lengths = torch.as_tensor(lengths, dtype=torch.int64)
#     lengths = lengths.cpu()
#     return PackedSequence(torch._C._VariableFunctions._pack_padded_sequence(input, lengths, batch_first))

def gru_forward(gru, input, lengths, state=None, batch_first=True):
    input_lengths, perm = torch.sort(lengths, descending=True)

    input = input[perm]
    if state is not None:
        state = state[perm].transpose(0, 1).contiguous()

    total_length=input.size(1)
    if not batch_first:
        input = input.transpose(0, 1)  # B x L x N -> L x B x N
    packed = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first)
    # packed = hotfix_pack_padded_sequence(embedded, input_lengths, batch_first)
    # self.gru.flatten_parameters()
    outputs, state = gru(packed, state)  # -> L x B x N * n_directions, 1, B, N
    outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=batch_first, total_length=total_length)  # unpack (back to padded)

    _, perm = torch.sort(perm, descending=False)
    if not batch_first:
        outputs = outputs.transpose(0, 1)
    outputs=outputs[perm]
    state = state.transpose(0, 1)[perm]

    return outputs, state

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.ne(PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    if torch.cuda.is_available():
        subsequent_mask=subsequent_mask.cuda()
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    subsequent_mask=1 - subsequent_mask

    return subsequent_mask