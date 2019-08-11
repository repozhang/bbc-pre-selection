import torch.nn as nn
import torch.nn.functional as F
from data.Utils import *
from EncDecModel import *
from modules.Criterions import *
from modules.Generators import *

class Environment(nn.Module):
    def __init__(self,context_size, action_size, hidden_size):
        super(Environment, self).__init__()

        self.c_enc = nn.GRU(context_size, hidden_size, num_layers=1, bidirectional=True, dropout=0.5, batch_first=True)
        self.a_enc = nn.GRU(action_size, hidden_size, num_layers=1, bidirectional=True, dropout=0.5, batch_first=True)

        self.attn = BilinearAttention(
            query_size=2*hidden_size, key_size=2*hidden_size, hidden_size=hidden_size, dropout=0.5, coverage=False
        )
        self.match_gru = nn.GRU(4*hidden_size, hidden_size, num_layers=1, bidirectional=True, dropout=0.5, batch_first=True)
        self.mlp=nn.Sequential(nn.Linear(2*hidden_size,1,bias=False), nn.Sigmoid())

    def reward(self, context, action, context_mask, action_mask):
        context,_=gru_forward(self.c_enc, context, context_mask.sum(dim=1).long())
        action, _ = gru_forward(self.a_enc, action, action_mask.sum(dim=1).long())

        action_att, _ = self.attn(action, context, context, query_mask=action_mask, key_mask=context_mask)
        action_mask = action_mask.float().detach()
        feature = torch.cat([action_att, action], dim=2) * action_mask.unsqueeze(2)
        feature, _ = gru_forward(self.match_gru, feature, action_mask.sum(dim=1).long())
        rewards = self.mlp(feature).squeeze(2) * action_mask

        return rewards

    def forward(self, context, action, context_mask, action_mask, y):
        rewards=self.reward(context, action, context_mask, action_mask)
        rewards = rewards.sum(dim=1) / action_mask.float().sum(dim=1)
        if y==1:
            gt = torch.ones(rewards.size(0)).float()
            if torch.cuda.is_available():
                gt = gt.cuda()
        elif y==0:
            gt = torch.zeros(rewards.size(0)).float()
            if torch.cuda.is_available():
                gt = gt.cuda()
        loss=F.binary_cross_entropy(rewards, gt)+1e-2*torch.distributions.bernoulli.Bernoulli(probs=rewards).entropy().mean()
        return loss

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embedding_size, hidden_size):
        super(Encoder, self).__init__()
        self.c_embedding = nn.Embedding(src_vocab_size, embedding_size, padding_idx=PAD)
        # self.b_embedding = nn.Embedding(src_vocab_size, embedding_size,padding_idx=PAD)
        self.b_embedding = self.c_embedding
        self.c_embedding_dropout = nn.Dropout(0.5)
        self.b_embedding_dropout = nn.Dropout(0.5)

        self.c_enc = nn.GRU(embedding_size, hidden_size, num_layers=1, bidirectional=True, dropout=0.5, batch_first=True)
        self.b_enc = nn.GRU(embedding_size, hidden_size, num_layers=1, bidirectional=True, dropout=0.5, batch_first=True)

        self.attn = BilinearAttention(
            query_size=2 * hidden_size, key_size=2 * hidden_size, hidden_size=hidden_size, dropout=0.5, coverage=False
        )

        self.matching_gru = nn.GRU(8 * hidden_size, hidden_size, num_layers=1, bidirectional=True, dropout=0.5)

    def forward(self, data):
        c_mask = data['context'].ne(PAD).detach()
        b_mask = data['background'].ne(PAD).detach()

        c_words = self.c_embedding_dropout(self.c_embedding(data['context']))
        b_words = self.b_embedding_dropout(self.b_embedding(data['background']))

        c_lengths = c_mask.sum(dim=1).detach()
        b_lengths = b_mask.sum(dim=1).detach()
        c_enc_output, c_state = gru_forward(self.c_enc, c_words, c_lengths)
        b_enc_output, b_state = gru_forward(self.b_enc, b_words, b_lengths)

        batch_size, c_len, hidden_size = c_enc_output.size()
        batch_size, b_len, hidden_size = b_enc_output.size()

        score = self.attn.unnormalized_score(b_enc_output, c_enc_output,
                                             key_mask=c_mask)  # batch_size, bg_len, src_len

        b2c = F.softmax(score, dim=2)
        b2c = b2c.masked_fill((1 - b_mask).unsqueeze(2).expand(batch_size, b_len, c_len), 0)
        b2c = torch.bmm(b2c, c_enc_output)  # batch_size, bg_len, hidden_size

        c2b = F.softmax(torch.max(score, dim=2)[0], dim=1).unsqueeze(1)  # batch_size, 1, bg_len
        c2b = torch.bmm(c2b, b_enc_output).expand(-1, b_len, -1)  # batch_size, bg_len, hidden_size

        g = torch.cat([b_enc_output, b2c, b_enc_output * b2c, b_enc_output * c2b], dim=-1)  # batch_size, bg_len, 8*hidden_size

        m, _ = gru_forward(self.matching_gru, g, b_mask.sum(dim=1))  # batch_size, bg_len, 2*hidden_size

        return c_enc_output, c_state, b_enc_output, b_state, g, m

class Selector(nn.Module):
    def __init__(self, embedding_size, hidden_size, tgt_vocab_size, embedding=None):
        super(Selector, self).__init__()

        if embedding is None:
            self.o_embedding = nn.Embedding(tgt_vocab_size, embedding_size, padding_idx=PAD)
        else:
            self.o_embedding =embedding
        self.o_embedding_dropout = nn.Dropout(0.5)

        self.matching_gru = nn.GRU(hidden_size+embedding_size+2*hidden_size, hidden_size, num_layers=1, bidirectional=True, dropout=0.5)

        self.p1_g = nn.Linear(8 * hidden_size, 1, bias=False)
        self.p1_m = nn.Linear(2*hidden_size, 1, bias=False)
        self.p1_s = nn.Linear(hidden_size+embedding_size+2*hidden_size, 1, bias=False)
        self.p1_t = nn.Linear(2*hidden_size, 1, bias=False)

    def forward(self, data, tgt, state, encode_output):
        b = data['background']
        b_mask = b.ne(PAD)
        c_enc_output, c_state, b_enc_output, b_state, g, m = encode_output
        batch_size, b_len, _=b_enc_output.size()

        embedded = self.o_embedding(tgt)
        embedded = self.o_embedding_dropout(embedded)

        s = torch.cat([state[0].expand(-1,b_len,-1), embedded.unsqueeze(1).expand(-1,b_len,-1), m], dim=-1)
        t, _ = gru_forward(self.matching_gru, s, b_mask.sum(dim=1))

        p1 = (self.p1_g(g) + self.p1_m(m)+ self.p1_s(s)+ self.p1_t(t)).squeeze(2)  # batch_size, bg_len
        p1 = p1.masked_fill(1 - b_mask, -float('inf'))

        return p1, state


class Generator(nn.Module):
    def __init__(self,embedding_size, hidden_size, tgt_vocab_size, embedding=None):
        super(Generator, self).__init__()

        self.embedding_size=embedding_size
        self.hidden_size=hidden_size

        if embedding is None:
            self.o_embedding = nn.Embedding(tgt_vocab_size, embedding_size, padding_idx=PAD)
        else:
            self.o_embedding =embedding
        self.o_embedding_dropout = nn.Dropout(0.5)

        self.attn = BilinearAttention(
            query_size=hidden_size, key_size=2*hidden_size, hidden_size=hidden_size, dropout=0.5, coverage=False
        )

        self.dec = nn.GRU(2*hidden_size + embedding_size, hidden_size, bidirectional=False, num_layers=1, dropout=0.5, batch_first=True)
        self.readout = nn.Linear(embedding_size + hidden_size + 2*hidden_size, hidden_size)
        self.gen=LinearGenerator(feature_size=hidden_size, tgt_vocab_size=tgt_vocab_size)

    def forward(self, data, tgt, state, encode_output):
        c_enc_output, c_state, b_enc_output, b_state, g, m=encode_output
        c_mask=data['context'].ne(PAD)

        state = state[0]

        embedded = self.o_embedding(tgt)
        embedded = self.o_embedding_dropout(embedded)

        attn_context_1, attn = self.attn(state, c_enc_output, c_enc_output, query_mask=None, key_mask=c_mask)

        gru_input = torch.cat((embedded.unsqueeze(1), attn_context_1), dim=2)
        gru_output, state = self.dec(gru_input, state.transpose(0,1))
        state=state.transpose(0,1)

        attn_context, attn = self.attn(state, c_enc_output, c_enc_output, query_mask=None, key_mask=c_mask)
        attn_context = attn_context.squeeze(1)

        concat_output = torch.cat((embedded, state.squeeze(1), attn_context), dim=1)

        feature_output = self.readout(concat_output)

        return self.gen(feature_output, softmax=False), [state]

class Mixture(nn.Module):
    def __init__(self, state_size):
        super(Mixture, self).__init__()
        self.linear_mixture = nn.Linear(state_size, 1)

    def forward(self, state,  selector_action, generator_action, b_map):
        p_s_g = torch.sigmoid(self.linear_mixture(state[0].squeeze(1)))

        selector_action=F.softmax(selector_action, dim=1)
        generator_action = F.softmax(generator_action, dim=1)

        generator_action = torch.mul(generator_action, p_s_g.expand_as(generator_action))

        selector_action = torch.bmm(selector_action.unsqueeze(1), b_map.float()).squeeze(1)

        selector_action = torch.mul(selector_action, (1-p_s_g).expand_as(selector_action))

        return torch.cat([generator_action, selector_action], 1)

class CaKe(EncDecModel):
    def __init__(self, encoder, selector, generator, env, src_id2vocab, src_vocab2id, tgt_id2vocab, tgt_vocab2id, max_dec_len, beam_width, eps=1e-10):
        super(CaKe, self).__init__(src_vocab_size=len(src_id2vocab), embedding_size=generator.embedding_size,
                                      hidden_size=generator.hidden_size, tgt_vocab_size=len(tgt_id2vocab), src_id2vocab=src_id2vocab,
                                      src_vocab2id=src_vocab2id, tgt_id2vocab=tgt_id2vocab, tgt_vocab2id=tgt_vocab2id, max_dec_len=max_dec_len, beam_width=beam_width,
                                      eps=eps)
        self.encoder=encoder
        self.selector=selector
        self.generator=generator
        self.env=env

        self.state_initializer = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.mixture=Mixture(self.hidden_size)
        self.criterion = CopyCriterion(len(tgt_id2vocab), force_copy=False, eps=eps)

    def encode(self,data):
        return self.encoder(data)

    def init_decoder_states(self,data, encode_output):
        c_enc_output, c_state, b_enc_output, b_state, g, m = encode_output
        batch_size=c_state.size(0)

        return [self.state_initializer(c_state.contiguous().view(batch_size,-1)).view(batch_size, 1, -1)]

    def decode(self, data, tgt, state, encode_output):
        sel_decode_output=self.selector(data, tgt, state, encode_output)
        gen_decode_output=self.generator(data, tgt, state, encode_output)
        return [sel_decode_output[0],gen_decode_output[0]], gen_decode_output[1]

    def generate(self, data, decode_output, softmax=True):
        actions, state = decode_output
        return self.mixture(state, actions[0], actions[1], data['background_map'])

    def loss(self,data, all_gen_output, all_decode_output, encode_output, reduction='mean'):
        loss=self.criterion(all_gen_output, data['output'], data['background_copy'], reduction=reduction)
        return loss
        # return loss+1e-2*torch.distributions.categorical.Categorical(probs=all_gen_output.view(-1, all_gen_output.size(2))).entropy().mean()

    def generation_to_decoder_input(self, data, indices):
        return indices.masked_fill(indices>=self.tgt_vocab_size, UNK)

    def to_word(self, data, gen_output, k=5, sampling=False):
        if not sampling:
            return copy_topk(gen_output, data['background_vocab_map'], data['background_vocab_overlap'], k=k)
        else:
            return randomk(gen_output, k=k)

    def to_sentence(self, data, batch_indices):
        return to_copy_sentence(data, batch_indices, self.tgt_id2vocab, data['background_dyn_vocab'])

    def forward(self, data, method='mle_train'):
        if method=='mle_train':
            return self.mle_train(data)
        elif method=='cake_train':
            return self.cake_train(data)
        elif method=='env_train':
            return self.env_train(data)
        elif method=='test':
            if self.beam_width==1:
                return self.greedy(data)
            else:
                return self.beam(data)
        elif method=='sample':
            return self.sample(data)

    def env_train(self, data):
        c_mask = data['context'].ne(PAD).detach()
        o_mask = data['output'].ne(PAD).detach()

        with torch.no_grad():
            c = self.encoder.c_embedding(data['context']).detach()
            o = self.generator.o_embedding(data['output']).detach()

            a, encode_outputs, init_decoder_states, all_decode_outputs, all_gen_outputs=sample(self, data, max_len=self.max_dec_len)
            a.masked_fill_(a >= self.tgt_vocab_size, UNK)
            a_mask = a.ne(PAD).detach()
            a = self.generator.o_embedding(a).detach()

        return self.env(c, o, c_mask, o_mask, 1).unsqueeze(0), self.env(c, a, c_mask, a_mask, 0).unsqueeze(0)

    def mle_train(self, data):
        encode_output, init_decoder_state, all_decode_output, all_gen_output=decode_to_end(self,data,schedule_rate=1)

        gen_loss=self.loss(data,all_gen_output,all_decode_output,encode_output).unsqueeze(0)

        return gen_loss
