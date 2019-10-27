from Constants import *
import torch
from modules.Attentions import *

class RecallCriterion(nn.Module):
    def __init__(self, hidden_size):
        super(RecallCriterion, self).__init__()
        self.hidden_size=hidden_size

    def forward(self, y1, y2, mask1, mask2, reduction='mean'):
        y = torch.ones(mask2.size())
        if torch.cuda.is_available():
            y = y.cuda()
        recall=(y1*mask1.float().unsqueeze(2)).sum(dim=1, keepdim=True).expand(-1, y2.size(1), -1).contiguous()
        r_loss = F.cosine_embedding_loss(y2.view(-1, self.hidden_size), recall.view(-1, self.hidden_size), y.view(-1), reduction='none')
        r_loss = r_loss * mask2.float().view(-1)
        r_loss = r_loss.sum() / mask2.sum().float().detach()

        return r_loss

class F1Criterion(nn.Module):
    def __init__(self, hidden_size):
        super(F1Criterion, self).__init__()
        self.hidden_size=hidden_size
        self.attn = BilinearAttention(
            query_size=hidden_size, key_size=hidden_size, hidden_size=hidden_size, dropout=0.5, coverage=False
        )

    def forward(self, y1, y2, mask1, mask2, reduction='mean'):
        y = torch.ones(mask1.size())
        if torch.cuda.is_available():
            y = y.cuda()
        precision,_=self.attn(y1, y2, y2)
        p_loss=F.cosine_embedding_loss(y1.view(-1, self.hidden_size), precision.view(-1, self.hidden_size), y.view(-1), reduction='none')
        p_loss= p_loss * mask1.float().view(-1)
        p_loss= p_loss.sum() / mask1.sum().float().detach()

        y = torch.ones(mask2.size())
        if torch.cuda.is_available():
            y = y.cuda()
        recall, _ = self.attn(y2, y1, y1)
        r_loss = F.cosine_embedding_loss(y2.view(-1, self.hidden_size), recall.view(-1, self.hidden_size), y.view(-1), reduction='none')
        r_loss = r_loss * mask2.float().view(-1)
        r_loss = r_loss.sum() / mask2.sum().float().detach()

        return (p_loss+r_loss)/2


class CopyCriterion(object):
    def __init__(self, tgt_vocab_size, force_copy=False, eps=1e-20):
        self.force_copy = force_copy
        self.eps = eps
        self.offset = tgt_vocab_size

    def __call__(self, gen_output, tgt, tgt_copy, reduction='mean'):
        copy_unk = tgt_copy.eq(UNK).float()
        copy_not_unk = tgt_copy.ne(UNK).float()
        target_unk = tgt.eq(UNK).float()
        target_not_unk = tgt.ne(UNK).float()
        target_not_pad=tgt.ne(PAD).float()

        # Copy probability of tokens in source
        if len(gen_output.size())>2:
            gen_output = gen_output.view(-1, gen_output.size(-1))
        copy_p = gen_output.gather(1, tgt_copy.view(-1, 1) + self.offset).view(-1)
        # Set scores for unk to 0 and add eps
        copy_p = copy_p.mul(copy_not_unk.view(-1)) + self.eps
        # Get scores for tokens in target
        tgt_p = gen_output.gather(1, tgt.view(-1, 1)).view(-1)

        # Regular prob (no unks and unks that can't be copied)
        if not self.force_copy:
            # Add score for non-unks in target
            p = copy_p + tgt_p.mul(target_not_unk.view(-1))
            # Add score for when word is unk in both align and tgt
            p = p + tgt_p.mul(copy_unk.view(-1)).mul(target_unk.view(-1))
        else:
            # Forced copy. Add only probability for not-copied tokens
            p = copy_p + tgt_p.mul(copy_unk.view(-1))

        p = p.log()

        # Drop padding.
        loss = -p.mul(target_not_pad.view(-1))
        if reduction=='mean':
            return loss.sum()/target_not_pad.sum()
        elif reduction=='none':
            return loss.view(tgt.size())

class CoverageCriterion(object):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def __call__(self, attentions, mask):

        sum_attentions=[attentions[0]]
        for i in range(len(attentions)-1):
            if i==0:
                sum_attentions.append(attentions[0])
            else:
                sum_attentions.append(sum_attentions[-1] + attentions[i])

        loss = torch.min(torch.cat(sum_attentions, dim=0), attentions[1:]).mul(mask.view(-1,1))

        loss = self.alpha*loss.sum()/loss.size(0)

        return loss