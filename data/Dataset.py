import codecs
from torch.utils.data import Dataset
from Constants import *
from data.Utils import *
import os
import re

class TrainDataset(Dataset):

    def __init__(self, files,src_vocab2id,tgt_vocab2id,srcs=None,tgts=None,max_src=80, max_tgt=40,n=1E10):
        super(TrainDataset, self).__init__()
        self.ids=list()
        self.srcs=list()
        self.tgts=list()
        self.src_vocab2id=src_vocab2id
        self.tgt_vocab2id=tgt_vocab2id
        self.files=files
        self.n=n
        self.max_src=max_src
        self.max_tgt=max_tgt
        if srcs is None:
            self.load()
        else:
            self.srcs=srcs
            self.tgts=tgts
            self.len=len(self.srcs)

    def load(self):
        id=0
        with codecs.open(self.files[0], encoding='utf-8') as f:
            for line in f:
                if len(self.srcs) >= self.n:
                    break
                temp = line.strip('\n').strip('\r').lower().replace('<s> ','')
                words = re.split('\s', temp)
                self.srcs.append(words[:min(len(words), self.max_src)])
                self.ids.append(id)
                id = id + 1
        with codecs.open(self.files[1], encoding='utf-8') as f:
            for line in f:
                if len(self.tgts) >= self.n:
                    break
                temp = line.strip('\n').strip('\r').lower().replace('<s> ','')
                words = re.split('\s', temp)
                self.tgts.append(words[:min(len(words), self.max_tgt)])
        self.len = len(self.srcs)
        print('data size: ', self.len)


    def __getitem__(self, index):
        src = torch.tensor([self.src_vocab2id.get(w, self.src_vocab2id.get(UNK_WORD)) for w in self.srcs[index]] + [
            self.src_vocab2id.get(EOS_WORD)], requires_grad=False).long()
        tgt = torch.tensor([self.tgt_vocab2id.get(w, self.tgt_vocab2id.get(UNK_WORD)) for w in self.tgts[index]] + [
            self.tgt_vocab2id.get(EOS_WORD)], requires_grad=False).long()
        return [self.ids[index], src, tgt]

    def __len__(self):
        return self.len

    def input(self,id):
        return self.srcs[id]

    def output(self,id):
        return self.tgts[id]

def train_collate_fn(data):
    id,src,tgt = zip(*data)

    src = merge1D(src)
    tgt = merge1D(tgt)
    id = torch.tensor(id, requires_grad=False)

    return {'id':id, 'input':src, 'output':tgt}


class TestDataset(Dataset):

    def __init__(self, files,src_vocab2id,tgt_vocab2id,srcs=None,tgts=None,max_src=80, n=1E10):
        super(TestDataset, self).__init__()
        self.ids=list()
        self.srcs = list()
        self.tgts = list()
        self.src_vocab2id=src_vocab2id
        self.tgt_vocab2id=tgt_vocab2id
        self.files=files
        self.n=n
        self.max_src=max_src
        if srcs is None:
            self.load()
        else:
            self.srcs=srcs
            self.tgts=tgts
            self.len=len(self.srcs)

    def load(self):
        id=0
        with codecs.open(self.files[0], encoding='utf-8') as f:
            for line in f:
                if len(self.srcs) >= self.n:
                    break
                temp = line.strip('\n').strip('\r').lower().replace('<s> ','')
                words = re.split('\s', temp)
                self.srcs.append(words[:min(len(words), self.max_src)])
                self.ids.append(id)
                id=id+1

        for file in self.files[1:]:
            temp_tgt=list()
            with codecs.open(file, encoding='utf-8') as f:
                for line in f:
                    if len(self.tgts)>=self.n:
                        break
                    temp = line.strip('\n').strip('\r').lower().replace('<s> ','')
                    temp_tgt.append(temp)
            if len(self.tgts)==0:
                for i in range(len(temp_tgt)):
                    self.tgts.append([os.linesep.join([sent.strip() for sent in re.split(r'<s>|</s>', temp_tgt[i])]).strip('\n').strip('\r')])
            else:
                for i in range(len(temp_tgt)):
                    self.tgts[i].append(os.linesep.join([sent.strip() for sent in re.split(r'<s>|</s>', temp_tgt[i])]).strip('\n').strip('\r'))

        self.len=len(self.srcs)
        print('data size: ',self.len)

    def __getitem__(self, index):
        src = torch.tensor([self.src_vocab2id.get(w, self.src_vocab2id.get(UNK_WORD)) for w in self.srcs[index]]+ [
            self.src_vocab2id.get(EOS_WORD)], requires_grad=False).long()

        return [self.ids[index], src]

    def __len__(self):
        return self.len

    def input(self,id):
        return self.srcs[id]

    def output(self,id):
        return self.tgts[id]

def test_collate_fn(data):
    id, src = zip(*data)

    src = merge1D(src)
    id = torch.tensor(id, requires_grad=False)

    return {'id':id, 'input':src}

