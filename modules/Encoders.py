import torch.nn as nn
from modules.Utils import *
from Constants import *
from modules.Attentions import *
from Utils import *

class GRUEncoder(nn.Module):
    def __init__(self, src_vocab_size, embedding_size, hidden_size, embedding_weight=None, num_layers=4, dropout=0.5):
        super(GRUEncoder, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.embedding_size=embedding_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers

        self.embedding = nn.Embedding(src_vocab_size, embedding_size, padding_idx=0, _weight=embedding_weight)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=self.num_layers, bidirectional=True, dropout=dropout, batch_first=True)


    def forward(self, src, state=None):
        embedded = self.embedding(src)
        embedded =self.embedding_dropout(embedded)

        outputs, state=gru_forward(self.gru, embedded, src.ne(PAD).sum(dim=1),state)

        return outputs, state