# coding: utf-8
import torch
import torch.nn.functional as F
import math
import random
from data.Utils import *
from torch.distributions.categorical import *
from Constants import *
from modules.Utils import *

def sample(model, data, max_len=20):
    batch_size = data['id'].size(0)

    encode_outputs = model.encode(data)

    init_decoder_states = model.init_decoder_states(data, encode_outputs)

    init_decoder_input = new_tensor([BOS] * batch_size, requires_grad=False)

    indices = list()
    end = new_tensor([0] * batch_size).long() == 1

    decoder_input = init_decoder_input
    decoder_states = init_decoder_states
    all_gen_outputs = list()
    all_decode_outputs = list()

    # ranp=random.randint(0, max_len-1)
    for t in range(max_len):
        decode_outputs = model.decode(
            data, decoder_input, decoder_states, encode_outputs
        )
        print(decode_outputs[2])

        gen_output = model.generate(data, decode_outputs, softmax=True)

        all_gen_outputs.append(gen_output.unsqueeze(0))
        all_decode_outputs.append(decode_outputs)

        probs, ids = model.to_word(data, F.softmax(gen_output, dim=1), 1, sampling=True)
        print(probs, ids)
        # if random.uniform(0,1)>0.9:
        #     probs, ids = model.to_word(data, F.softmax(gen_output, dim=1), 1, sampling=True)
        # else:
        #     probs, ids = model.to_word(data, F.softmax(gen_output, dim=1), 1, sampling=False)

        decoder_states = decode_outputs[1]

        indice = ids[:, 0]
        this_end = indice == EOS
        if t == 0:
            indice.masked_fill_(this_end, UNK)
        elif t==max_len-1:
            indice[:]=EOS
            indice.masked_fill_(end, PAD)
        else:
            indice.masked_fill_(end, PAD)
        indices.append(indice.unsqueeze(1))
        end = end | this_end

        decoder_input = model.generation_to_decoder_input(data, indice)

    all_gen_outputs = torch.cat(all_gen_outputs, dim=0).transpose(0, 1).contiguous()

    return torch.cat(indices, dim=1), encode_outputs, init_decoder_states, all_decode_outputs, all_gen_outputs


def greedy(model,data,max_len=20):
    batch_size=data['id'].size(0)

    encode_outputs= model.encode(data)

    decoder_states = model.init_decoder_states(data, encode_outputs)

    decoder_input = new_tensor([BOS] * batch_size, requires_grad=False)

    greedy_indices=list()
    greedy_end = new_tensor([0] * batch_size).long() == 1
    for t in range(max_len):
        decode_outputs = model.decode(
            data, decoder_input, decoder_states, encode_outputs
        )

        gen_output=model.generate(data, decode_outputs, softmax=True)

        probs, ids=model.to_word(data, gen_output, 1)

        decoder_states = decode_outputs[1]

        greedy_indice = ids[:,0]
        greedy_this_end = greedy_indice == EOS
        if t == 0:
            greedy_indice.masked_fill_(greedy_this_end, UNK)
        else:
            greedy_indice.masked_fill_(greedy_end, PAD)
        greedy_indices.append(greedy_indice.unsqueeze(1))
        greedy_end = greedy_end | greedy_this_end

        decoder_input = model.generation_to_decoder_input(data, greedy_indice)

    greedy_indice=torch.cat(greedy_indices,dim=1)
    return greedy_indice


def beam(model,data,max_len=20,width=5):
    batch_size = data['id'].size(0)

    encode_outputs = model.encode(data)
    decoder_states = model.init_decoder_states(data, encode_outputs)

    num_states=len(decoder_states)
    num_encodes=len(encode_outputs)

    next_fringe = []
    results = dict()
    for i in range(batch_size):
        next_fringe += [Node(parent=None, state=[s[i].unsqueeze(0) for s in decoder_states], word=BOS_WORD, value=BOS, cost=0.0, encode_outputs=[o[i].unsqueeze(0) for o in encode_outputs], data=get_data(i,data), batch_id=i)]
        results[i] = []

    for l in range(max_len+1):
        fringe = []
        for n in next_fringe:
            if n.value == EOS or l == max_len:
                results[n.batch_id].append(n)
            else:
                fringe.append(n)

        if len(fringe) == 0:
            break

        data=concat_data([n.data for n in fringe])

        decoder_input= new_tensor([n.value for n in fringe], requires_grad=False)
        decoder_input = model.generation_to_decoder_input(data, decoder_input)

        decoder_states=[]
        for i in range(num_states):
            decoder_states+=[torch.cat([n.state[i] for n in fringe],dim=0)]

        encode_outputs=[]
        for i in range(num_encodes):
            encode_outputs+=[torch.cat([n.encode_outputs[i] for n in fringe],dim=0)]

        decode_outputs = model.decode(
            data, decoder_input, decoder_states, encode_outputs
        )
        decoder_states = decode_outputs[1]

        gen_output = model.generate(data, decode_outputs, softmax=True)

        probs, ids = model.to_word(data, gen_output, width)

        next_fringe_dict = dict()
        for i in range(batch_size):
            next_fringe_dict[i] = []

        for i in range(len(fringe)):
            n = fringe[i]
            state_n = [s[i].unsqueeze(0) for s in decoder_states]

            for j in range(width):
                loss = -math.log(probs[i,j].item() + 1e-10)

                n_new = Node(parent=n, state=state_n, word=None, value=ids[i,j].item(), cost=loss,
                             encode_outputs=n.encode_outputs,
                             data=n.data, batch_id=n.batch_id)

                next_fringe_dict[n_new.batch_id].append(n_new)

        next_fringe = []
        for i in range(batch_size):
            next_fringe += sorted(next_fringe_dict[i], key=lambda n: n.cum_cost / n.length)[:width]

    outputs = []
    for i in range(batch_size):
        results[i].sort(key=lambda n: n.cum_cost / n.length)
        outputs.append(results[i][0])# currently only select the first one

    # sents=[node.to_sequence_of_words()[1:-1] for node in outputs]
    indices=merge1D([new_tensor(node.to_sequence_of_values()[1:]) for node in outputs])

    return indices



def transformer_greedy(model,data,max_len=20):
    batch_size = data['id'].size(0)

    encode_outputs= model.encode(data)

    decoder_input = new_tensor([BOS] * batch_size).view(batch_size, 1)

    greedy_indices=list()
    greedy_indices.append(decoder_input)
    greedy_end = new_tensor([0] * batch_size).long() == 1
    for t in range(max_len):
        decoder_input = torch.cat(greedy_indices, dim=1)
        decode_outputs = model.decode(
            data, decoder_input, None, encode_outputs
        )

        batch_size, tgt_len, hidden_size=decode_outputs.size()
        decode_outputs = decode_outputs.view(batch_size, tgt_len, -1)[:, -1]
        gen_output=model.generate(data, decode_outputs, softmax=True)

        probs, ids=model.to_word(data, gen_output, 5)

        greedy_indice = ids[:,2]
        greedy_this_end = greedy_indice == EOS
        if t == 0:
            greedy_indice.masked_fill_(greedy_this_end, UNK)
        else:
            greedy_indice.masked_fill_(greedy_end, PAD)
        greedy_indices.append(greedy_indice.unsqueeze(1))
        greedy_end = greedy_end | greedy_this_end

    greedy_indice=torch.cat(greedy_indices,dim=1)
    return greedy_indice


def transformer_beam(model,data,max_len=20,width=5):
    batch_size = data['id'].size(0)

    encode_outputs = model.encode(data)

    num_encodes=len(encode_outputs)

    next_fringe = []
    results = dict()
    for i in range(batch_size):
        next_fringe += [Node(parent=None, state=None, word=BOS_WORD, value=BOS, cost=0.0, encode_outputs=[o[i].unsqueeze(0) for o in encode_outputs], data=get_data(i,data), batch_id=i)]
        results[i] = []

    for l in range(max_len+1):
        fringe = []
        for n in next_fringe:
            if n.value == EOS or l == max_len:
                results[n.batch_id].append(n)
            else:
                fringe.append(n)

        if len(fringe) == 0:
            break

        decoder_input = merge1D([new_tensor(node.to_sequence_of_values()) for node in fringe])
        decoder_input = model.generation_to_decoder_input(data, decoder_input)

        encode_outputs=[]
        for i in range(num_encodes):
            encode_outputs+=[torch.cat([n.encode_outputs[i] for n in fringe],dim=0)]

        data=concat_data([n.data for n in fringe])

        decode_outputs = model.decode(
            data, decoder_input, None, encode_outputs
        )

        this_batch_size, tgt_len, hidden_size = decode_outputs.size()
        lengths = decoder_input.ne(PAD).sum(dim=1).long() - 1
        last_decode_output = list()
        for i in range(this_batch_size):
            last_decode_output.append(decode_outputs[i, lengths[i].item()].unsqueeze(0))
        decode_outputs = torch.cat(last_decode_output, dim=0)

        gen_output = model.generate(data, decode_outputs, softmax=True)

        probs, ids = model.to_word(data, gen_output, width)

        next_fringe_dict = dict()
        for i in range(batch_size):
            next_fringe_dict[i] = []

        for i in range(len(fringe)):
            n = fringe[i]

            for j in range(width):
                loss = -math.log(probs[i,j].item() + 1e-10)

                n_new = Node(parent=n, state=None, word=None, value=ids[i,j].item(), cost=loss,
                             encode_outputs=n.encode_outputs,
                             data=n.data, batch_id=n.batch_id)

                next_fringe_dict[n_new.batch_id].append(n_new)

        next_fringe = []
        for i in range(batch_size):
            next_fringe += sorted(next_fringe_dict[i], key=lambda n: n.cum_cost / n.length)[:width]

    outputs = []
    for i in range(batch_size):
        results[i].sort(key=lambda n: n.cum_cost / n.length)
        outputs.append(results[i][0])# currently only select the first one

    # sents=[node.to_sequence_of_words()[1:-1] for node in outputs]
    indices=merge1D([new_tensor(node.to_sequence_of_values()[1:]) for node in outputs])

    return indices

class Node(object):
    def __init__(self, parent, state, word, value, cost, encode_outputs, data, batch_id=None):
        super(Node, self).__init__()
        self.word=word
        self.value = value
        self.parent = parent # parent Node, None for root
        self.state = state
        self.cum_cost = parent.cum_cost + cost if parent else cost # e.g. -log(p) of sequence up to current node (including)
        self.length = 1 if parent is None else parent.length + 1
        self.encode_outputs = encode_outputs # can hold, for example, attention weights
        self._sequence = None
        self.batch_id=batch_id
        self.data=data

    def to_sequence(self):
        # Return sequence of nodes from root to current node.
        if not self._sequence:
            self._sequence = []
            current_node = self
            while current_node:
                self._sequence.insert(0, current_node)
                current_node = current_node.parent
        return self._sequence

    def to_sequence_of_values(self):
        return [s.value for s in self.to_sequence()]

    def to_sequence_of_words(self):
        return [s.word for s in self.to_sequence()]
