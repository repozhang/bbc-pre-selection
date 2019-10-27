from Constants import *
from data.Dataset import *

class CopyTrainDataset(TrainDataset):

    def __init__(self, files,src_vocab2id,tgt_vocab2id,srcs=None,tgts=None,max_src=80, max_tgt=40,n=1E10):
        super(CopyTrainDataset, self).__init__(files,src_vocab2id,tgt_vocab2id,srcs=srcs,tgts=tgts,max_src=max_src,max_tgt=max_tgt,n=n)

    def __getitem__(self, index):
        src = torch.tensor([self.src_vocab2id.get(w, self.src_vocab2id.get(UNK_WORD)) for w in self.srcs[index]],
                                  requires_grad=False).long()
        tgt = torch.tensor([self.tgt_vocab2id.get(w, self.tgt_vocab2id.get(UNK_WORD)) for w in self.tgts[index]] + [
            self.tgt_vocab2id.get(EOS_WORD)], requires_grad=False).long()
        dyn_vocab2id, dyn_id2vocab = build_vocab(self.srcs[index])

        src_copy = torch.tensor([dyn_vocab2id.get(w, dyn_vocab2id.get(UNK_WORD)) for w in self.tgts[index]] + [
            dyn_vocab2id.get(EOS_WORD)], requires_grad=False).long()
        src_map = build_words_vocab_map(self.srcs[index], dyn_vocab2id)

        vocab_map, vocab_overlap = build_vocab_vocab_map(dyn_id2vocab, self.tgt_vocab2id)

        return [self.ids[index],src, tgt, src_copy, src_map,dyn_id2vocab, vocab_map, vocab_overlap]


def train_collate_fn(data):
    id,src,tgt, tgt_copy, src_map,dyn_id2vocab, vocab_map, vocab_overlap = zip(*data)

    dyn_id2vocab_map={}
    for i in range(len(id)):
        dyn_id2vocab_map[id[i]]=dyn_id2vocab[i]

    src = merge1D(src)
    tgt = merge1D(tgt)
    tgt_copy = merge1D(tgt_copy)
    src_map = merge2D(src_map)
    id = torch.tensor(id, requires_grad=False)
    vocab_map = merge2D(vocab_map)
    vocab_overlap = merge1D(vocab_overlap)

    return {'id':id, 'input':src, 'output':tgt, 'input_copy':tgt_copy, 'input_map':src_map, 'input_dyn_vocab':dyn_id2vocab_map, 'vocab_map':vocab_map, 'vocab_overlap':vocab_overlap}


class CopyTestDataset(TestDataset):

    def __init__(self, files,src_vocab2id,tgt_vocab2id,srcs=None,tgts=None,max_src=80, n=1E10):
        super(CopyTestDataset, self).__init__(files,src_vocab2id,tgt_vocab2id,srcs=srcs,tgts=tgts,max_src=max_src,n=n)

    def __getitem__(self, index):
        src = torch.tensor([self.src_vocab2id.get(w, self.src_vocab2id.get(UNK_WORD)) for w in self.srcs[index]],
                                  requires_grad=False).long()
        dyn_vocab2id, dyn_id2vocab = build_vocab(self.srcs[index])

        src_map = build_words_vocab_map(self.srcs[index], dyn_vocab2id)

        vocab_map, vocab_overlap = build_vocab_vocab_map(dyn_id2vocab, self.tgt_vocab2id)

        return [self.ids[index], src, src_map, dyn_id2vocab, vocab_map, vocab_overlap]

def test_collate_fn(data):
    id, src, src_map,dyn_id2vocab, vocab_map, vocab_overlap = zip(*data)

    dyn_id2vocab_map={}
    for i in range(len(id)):
        dyn_id2vocab_map[id[i]]=dyn_id2vocab[i]

    src = merge1D(src)
    src_map = merge2D(src_map)
    id = torch.tensor(id, requires_grad=False)
    vocab_map=merge2D(vocab_map)
    vocab_overlap = merge1D(vocab_overlap)

    return {'id':id, 'input':src,'input_map':src_map, 'input_dyn_vocab':dyn_id2vocab_map, 'vocab_map':vocab_map, 'vocab_overlap':vocab_overlap}

