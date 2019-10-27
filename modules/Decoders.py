import torch
import torch.nn as nn
from modules.Attentions import *
from modules.Utils import *
from Constants import *
from modules.PositionwiseFeedForward import *
from Utils import *

class GruDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, tgt_vocab_size, embedding=None, num_layers=4, dropout=0.5):
        super(GruDecoder, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.embedding_size=embedding_size
        self.tgt_vocab_size = tgt_vocab_size

        if embedding is not None:
            self.embedding =embedding
        else:
            self.embedding = nn.Embedding(tgt_vocab_size, embedding_size, padding_idx=PAD)
        self.embedding_dropout = nn.Dropout(dropout)

        self.attn = BilinearAttention(
            query_size=hidden_size, key_size=2*hidden_size, hidden_size=hidden_size, dropout=dropout, coverage=False
        )

        self.gru = nn.GRU(2*hidden_size+embedding_size, hidden_size, bidirectional=False, num_layers=num_layers, dropout=dropout)

        self.readout = nn.Linear(embedding_size + hidden_size + 2*hidden_size, hidden_size)

    def forward(self, tgt, state, enc_output, enc_mask=None):
        gru_state = state[0]
        # sum_attention= state[1]

        embedded = self.embedding(tgt)
        embedded = self.embedding_dropout(embedded)

        attn_context_1, attn=self.attn(gru_state[:,-1].unsqueeze(1), enc_output, enc_output, query_mask=None, key_mask=enc_mask)
        attn_context_1=attn_context_1.squeeze(1)

        gru_input = torch.cat((embedded, attn_context_1), dim=1)
        gru_output, gru_state=self.gru(gru_input.unsqueeze(0), gru_state.transpose(0,1))
        # gru_output=gru_output.squeeze(0)
        gru_state=gru_state.transpose(0,1)

        attn_context, attn=self.attn(gru_state[:,-1].unsqueeze(1), enc_output, enc_output, query_mask=None, key_mask=enc_mask, sum_attention=None)
        attn=attn.squeeze(1)
        # sum_attention=sum_attention+attn
        attn_context = attn_context.squeeze(1)

        concat_output = torch.cat((embedded, gru_state[:,-1], attn_context), dim=1)

        feature_output=self.readout(concat_output)
        return feature_output, [gru_state], attn