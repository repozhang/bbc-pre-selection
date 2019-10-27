import torch
import torch.nn as nn
from modules.Attentions import *
from modules.Utils import *
from Constants import *

class BBCDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, tgt_vocab_size, embedding=None, num_layers=4, dropout=0.5):
        super(BBCDecoder, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.embedding_size=embedding_size
        self.tgt_vocab_size = tgt_vocab_size

        if embedding is not None:
            self.embedding =embedding
        else:
            self.embedding = nn.Embedding(tgt_vocab_size, embedding_size, padding_idx=PAD)
        self.embedding_dropout = nn.Dropout(dropout)

        self.src_attn = BilinearAttention(
            query_size=hidden_size, key_size=2*hidden_size, hidden_size=hidden_size, dropout=dropout, coverage=False
        )
        self.bg_attn = BilinearAttention(
            query_size=hidden_size, key_size=2 * hidden_size, hidden_size=hidden_size, dropout=dropout, coverage=False
        ) #background attention score

        self.gru = nn.GRU(2*hidden_size+2*hidden_size+embedding_size, hidden_size, bidirectional=False, num_layers=num_layers, dropout=dropout)

        self.readout = nn.Linear(embedding_size + hidden_size + 2*hidden_size+ 2*hidden_size, hidden_size)

    def forward(self, tgt, state, src_output, bg_output, src_mask=None, bg_mask=None):
        gru_state = state[0]

        embedded = self.embedding(tgt)
        embedded = self.embedding_dropout(embedded)

        src_context, src_attn=self.src_attn(gru_state[:,-1].unsqueeze(1), src_output, src_output, query_mask=None, key_mask=src_mask)
        src_context=src_context.squeeze(1)
        src_attn = src_attn.squeeze(1)
        #background attention, bg_context(context),bc_attn(attention score)
        bg_context, bg_attn = self.bg_attn(gru_state[:, -1].unsqueeze(1), bg_output, bg_output, query_mask=None, key_mask=bg_mask)
        bg_context = bg_context.squeeze(1)
        bg_attn = bg_attn.squeeze(1)

        #output bg_attn, score r_t, regularize r_t-r_{t-1}
        print('bg_attn',bg_attn.squeeze(0).tolist())

        gru_input = torch.cat((embedded, src_context, bg_context), dim=1)
        gru_output, gru_state=self.gru(gru_input.unsqueeze(0), gru_state.transpose(0,1))
        gru_state=gru_state.transpose(0,1)

        concat_output = torch.cat((embedded, gru_state[:,-1], src_context, bg_context), dim=1)

        feature_output=self.readout(concat_output)
        return feature_output, [gru_state], [src_attn, bg_attn], bg_context  #bg_attn: BBCDecoder[2][1]

class MALDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, tgt_vocab_size, embedding=None, num_layers=4, dropout=0.5):
        super(MALDecoder, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.embedding_size=embedding_size
        self.tgt_vocab_size = tgt_vocab_size

        if embedding is not None:
            self.embedding =embedding
        else:
            self.embedding = nn.Embedding(tgt_vocab_size, embedding_size, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)

        self.src_attn = BilinearAttention(
            query_size=hidden_size, key_size=2*hidden_size, hidden_size=hidden_size, dropout=dropout, coverage=False
        )
        self.bg_attn = BilinearAttention(
            query_size=hidden_size, key_size=2 * hidden_size, hidden_size=hidden_size, dropout=dropout, coverage=False
        )

        self.gru = nn.GRU(2*hidden_size+2*hidden_size+2*hidden_size+2*hidden_size+embedding_size, hidden_size, bidirectional=False, num_layers=num_layers, dropout=dropout)

        self.readout = nn.Linear(embedding_size + hidden_size + 2*hidden_size+ 2*hidden_size, hidden_size)

    def forward(self, tgt, state, src_output, bg_output, sel_bg, src_mask=None, bg_mask=None):
        gru_state = state[0]

        embedded = self.embedding(tgt)
        embedded = self.embedding_dropout(embedded)

        src_context, src_attn=self.src_attn(gru_state[:,-1].unsqueeze(1), src_output, src_output, query_mask=None, key_mask=src_mask)
        src_context=src_context.squeeze(1)
        src_attn = src_attn.squeeze(1)
        bg_context, bg_attn = self.bg_attn(gru_state[:, -1].unsqueeze(1), bg_output, bg_output, query_mask=None, key_mask=bg_mask)
        bg_context = bg_context.squeeze(1)
        bg_attn = bg_attn.squeeze(1)

        gru_input = torch.cat((embedded, src_context, bg_context, sel_bg), dim=1)
        gru_output, gru_state=self.gru(gru_input.unsqueeze(0), gru_state.transpose(0,1))
        gru_state=gru_state.transpose(0,1)

        concat_output = torch.cat((embedded, gru_state[:,-1], src_context, bg_context), dim=1)

        feature_output=self.readout(concat_output)
        return feature_output, [gru_state], [src_attn, bg_attn], bg_context