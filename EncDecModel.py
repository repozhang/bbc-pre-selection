import torch.nn as nn
from Utils import *
from modules.Generations import *

class EncDecModel(nn.Module):
    def __init__(self, src_vocab_size, embedding_size, hidden_size, tgt_vocab_size, src_id2vocab=None, src_vocab2id=None, tgt_id2vocab=None, tgt_vocab2id=None, max_dec_len=120, beam_width=1, eps=1e-10):
        super(EncDecModel, self).__init__()
        self.src_vocab_size=src_vocab_size
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.tgt_vocab_size=tgt_vocab_size
        self.tgt_id2vocab=tgt_id2vocab
        self.tgt_vocab2id=tgt_vocab2id
        self.src_id2vocab=src_id2vocab
        self.src_vocab2id=src_vocab2id
        self.eps=eps
        self.beam_width=beam_width
        self.max_dec_len=max_dec_len

    def encode(self, data):
        raise NotImplementedError

    def init_decoder_states(self, data, encode_output):
        return None

    def decode(self, data, tgt, state, encode_output):
        raise NotImplementedError

    def generate(self, data, decode_output, softmax=False):
        raise NotImplementedError

    def to_word(self, data, gen_output, k=5, sampling=False):
        if not sampling:
            return topk(gen_output, k=k)
        else:
            return randomk(gen_output, k=k)

    def generation_to_decoder_input(self, data, indices):
        return indices

    def loss(self,data, all_gen_output, all_decode_output, encode_output, reduction='mean'):
        raise NotImplementedError

    def to_sentence(self, data, batch_indice):
        return to_sentence(batch_indice, self.tgt_id2vocab)

    def sample(self, data):
        return sample(self, data, self.max_dec_len)

    def greedy(self, data):
        return greedy(self,data, self.max_dec_len)

    def beam(self, data):
        return beam(self, data, self.max_dec_len, self.beam_width)

    def mle_train(self, data):
        encode_output, init_decoder_state, all_decode_output, all_gen_output=decode_to_end(self,data,schedule_rate=1)

        loss=self.loss(data,all_gen_output,all_decode_output,encode_output)

        return loss.unsqueeze(0)

    def forward(self, data, method):
        if method=='mle_train':
            return self.mle_train(data)
        elif method=='reinforce_train':
            return self.reinforce_train(data)
        elif method=='test':
            if self.beam_width==1:
                return self.greedy(data)
            else:
                return self.beam(data)
        elif method=='sample':
            return self.sample(data)