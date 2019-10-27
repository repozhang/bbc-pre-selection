import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class BilinearAttention(nn.Module):
    def __init__(self, query_size, key_size, hidden_size, dropout=0.5, coverage=False, gumbel=False, temperature=100):
        super().__init__()
        self.linear_key = nn.Linear(key_size, hidden_size, bias=False)
        self.linear_query = nn.Linear(query_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.hidden_size=hidden_size
        # self.dropout = nn.Dropout(dropout)
        # self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        if coverage:
            self.linear_coverage = nn.Linear(1, hidden_size, bias=False)
        self.gumbel=gumbel
        self.temperature=temperature

    def score(self, query, key, query_mask=None, key_mask=None, mask=None, sum_attention=None):
        batch_size, key_len, key_size = key.size()
        batch_size, query_len, query_size = query.size()
        attn=self.unnormalized_score(query, key, key_mask, mask, sum_attention)

        # attn=self.softmax(attn.view(-1, key_len)).view(batch_size, query_len, key_len)
        if self.gumbel:
            if self.training:
                attn =F.gumbel_softmax(attn.view(attn.size(0),-1), self.temperature, hard=False).view(attn.size())
            else:
                attn_ = torch.zeros(attn.size())
                if torch.cuda.is_available():
                    attn_ = attn_.cuda()
                attn = attn_.scatter_(2, attn.argmax(dim=2).unsqueeze(2), 1)
        else:
            attn=F.softmax(attn, dim=2)
        if query_mask is not None:
            attn = attn.masked_fill(1-query_mask.unsqueeze(2).expand(batch_size, query_len, key_len), 0)
        # attn = self.dropout(attn)

        return attn


    def unnormalized_score(self, query, key, key_mask=None, mask=None, sum_attention=None):
        batch_size, key_len, key_size = key.size()
        batch_size, query_len, query_size = query.size()

        wq = self.linear_query(query.view(-1, query_size))
        wq = wq.view(batch_size, query_len, 1, self.hidden_size)
        wq = wq.expand(batch_size, query_len, key_len, self.hidden_size)

        uh = self.linear_key(key.view(-1, key_size))
        uh = uh.view(batch_size, 1, key_len, self.hidden_size)
        uh = uh.expand(batch_size, query_len, key_len, self.hidden_size)

        wuc = wq + uh
        if sum_attention is not None:
            batch_size, key_len=sum_attention.size()
            wc = self.linear_coverage(sum_attention.view(-1,1)).view(batch_size, 1, key_len, self.hidden_size)
            wc = wc.expand(batch_size, query_len, key_len, self.hidden_size)
            wuc = wuc + wc

        wquh = self.tanh(wuc)

        attn = self.v(wquh.view(-1, self.hidden_size)).view(batch_size, query_len, key_len)

        if key_mask is not None:
            attn = attn.masked_fill(1-key_mask.unsqueeze(1).expand(batch_size, query_len, key_len), -float('inf'))

        if mask is not None:
            attn = attn.masked_fill(1 - mask, -float('inf'))
        return attn

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None, sum_attention=None):

        attn = self.score(query, key, query_mask=query_mask, key_mask=key_mask, mask=mask, sum_attention=sum_attention)
        attn_value = torch.bmm(attn,value)

        return attn_value, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, dropout=0.5):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(dropout)
        # self.softmax = nn.Softmax(dim=1)

    def score(self, query, key, query_mask=None, key_mask=None, mask=None):
        batch_size, key_len, key_size = key.size()
        batch_size, query_len, query_size = query.size()
        attn = self.unnormalized_score(query, key, key_mask, mask)

        # attn = self.softmax(attn.view(-1, key_len)).view(batch_size, query_len, key_len)
        attn=F.softmax(attn, dim=2)
        if query_mask is not None:
            attn = attn.masked_fill(1 - query_mask.unsqueeze(2).expand(batch_size, query_len, key_len), 0)
        # attn = self.dropout(attn)
        return attn

    def unnormalized_score(self, query, key, key_mask=None, mask=None):
        batch_size, key_len, key_size = key.size()
        batch_size, query_len, query_size = query.size()

        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / self.temperature

        if key_mask is not None:
            attn = attn.masked_fill(1-key_mask.unsqueeze(1).expand(batch_size, query_len, key_len), -float('inf'))

        if mask is not None:
            attn = attn.masked_fill(1 - mask, -float('inf'))

        return attn

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        attn = self.score(query, key, query_mask=query_mask, key_mask=key_mask, mask=mask)
        attn_value = torch.bmm(attn, value)  # attn: attention weight; value: vectors

        return attn_value, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, num_head, query_size, key_size, value_size, dropout=0.5, attention=None):
        super().__init__()

        self.num_head = num_head
        self.key_size = key_size
        self.value_size = value_size

        self.w_qs = nn.Linear(query_size, num_head * query_size)
        self.w_ks = nn.Linear(key_size, num_head * key_size)
        self.w_vs = nn.Linear(value_size, num_head * value_size)

        if attention is None:
            self.attention = ScaledDotProductAttention(temperature=np.power(key_size, 0.5))
        else:
            self.attention =attention

        self.layer_norm = nn.LayerNorm(query_size)

        self.fc = nn.Linear(num_head * value_size, query_size)

        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):


        batch_size, query_len, query_size = query.size()
        batch_size, key_len, key_size = key.size()
        batch_size, value_len, value_size = value.size()

        num_head=  self.num_head

        residual = query

        query = self.w_qs(query).view(batch_size, query_len, num_head, query_size)
        key = self.w_ks(key).view(batch_size, key_len, num_head, key_size)
        value = self.w_vs(value).view(batch_size, value_len, num_head, value_size)

        query = query.permute(2, 0, 1, 3).contiguous().view(-1, query_len, query_size)
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, key_len, key_size)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, value_len, value_size)

        if query_mask is not None:
            query_mask = query_mask.unsqueeze(0).expand(num_head, batch_size, query_len).contiguous().view(-1,query_len)
        if key_mask is not None:
            # print(key_mask.size(), key_len)
            key_mask = key_mask.unsqueeze(0).expand(num_head, batch_size, key_len).contiguous().view(-1,key_len)
        if mask is not None:
            mask = mask.unsqueeze(0).expand(num_head, batch_size, query_len, key_len).contiguous().view(-1, query_len, key_len)

        output, attn = self.attention(query, key, value, query_mask=query_mask, key_mask=key_mask, mask=mask)

        output = output.view(num_head, batch_size, query_len, value_size)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, query_len, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn.view(num_head, batch_size, query_len, key_len)
