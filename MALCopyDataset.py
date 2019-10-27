import codecs
from torch.utils.data import Dataset
from Constants import *
from data.Utils import *
import json

# from multiprocessing.dummy import Pool as ThreadPool
# pool = ThreadPool(32)

class MALDataset(Dataset):
    def __init__(self, files, src_vocab2id, tgt_vocab2id,n=1E10):
        super(MALDataset, self).__init__()
        self.ids = list()
        self.contexts = list()
        self.queries = list()
        self.outputs = list()
        self.backgrounds = list()

        self.id_arrays = list()
        self.context_arrays = list()
        self.query_arrays = list()
        self.output_arrays = list()
        self.background_arrays = list()
        self.background_selection_arrays = list()
        self.background_ref_start_arrays = list()
        self.background_ref_end_arrays = list()

        self.bg_dyn_vocab2ids=list()
        self.bg_dyn_id2vocabs=list()
        self.background_copy_arrays= list()

        self.src_vocab2id=src_vocab2id
        self.tgt_vocab2id=tgt_vocab2id
        self.files=files
        self.n=n

        self.load()

    def load(self):
        with codecs.open(self.files[0], encoding='utf-8') as f:
            data = json.load(f)
            for id in range(len(data)):
                sample=data[id]

                context = sample['context'].split(' ')
                self.contexts.append(context)
                self.context_arrays.append(torch.tensor([self.src_vocab2id.get(w.lower(), UNK) for w in context], requires_grad=False).long())

                query = sample['query'].split(' ')
                # query = context
                # query = query[max(0, len(query) - 65):]
                self.queries.append(query)
                self.query_arrays.append(torch.tensor([self.src_vocab2id.get(w.lower(), UNK) for w in query], requires_grad=False).long())

                background = sample['background'].split(' ')
                # background = background[:min(self.max_bg, len(background))]
                self.backgrounds.append(background)
                self.background_arrays.append(torch.tensor([self.src_vocab2id.get(w.lower(), UNK) for w in background], requires_grad=False).long())

                bg_dyn_vocab2id, bg_dyn_id2vocab = build_vocab(sample['background'].lower().split(' '))
                self.bg_dyn_vocab2ids.append((id, bg_dyn_vocab2id))
                self.bg_dyn_id2vocabs.append((id, bg_dyn_id2vocab))

                output = sample['response'].lower().split(' ')
                self.outputs.append(output)
                self.output_arrays.append(torch.tensor([self.tgt_vocab2id.get(w, UNK) for w in output] + [EOS], requires_grad=False).long())
                self.background_copy_arrays.append(torch.tensor([bg_dyn_vocab2id.get(w, UNK) for w in output] + [EOS], requires_grad=False).long())

                output = set(output)
                self.background_selection_arrays.append(torch.tensor([1 if w.lower() in output else 0 for w in background], requires_grad=False).long())

                if 'bg_ref_start' in sample:
                    self.background_ref_start_arrays.append(torch.tensor([sample['bg_ref_start']], requires_grad=False))
                    self.background_ref_end_arrays.append(torch.tensor([sample['bg_ref_end'] - 1], requires_grad=False))
                else:
                    self.background_ref_start_arrays.append(torch.tensor([-1], requires_grad=False))
                    self.background_ref_end_arrays.append(torch.tensor([-1], requires_grad=False))

                self.ids.append(id)
                self.id_arrays.append(torch.tensor([id]).long())

                if len(self.contexts)>=self.n:
                    break
        self.len = len(self.contexts)
        print('data size: ', self.len)

    def __getitem__(self, index):
        return [self.ids[index], self.id_arrays[index], self.contexts[index],self.context_arrays[index],self.queries[index], self.query_arrays[index],
                self.backgrounds[index], self.background_arrays[index], self.outputs[index], self.output_arrays[index],self.background_copy_arrays[index],
                self.bg_dyn_id2vocabs[index], self.bg_dyn_vocab2ids[index], self.tgt_vocab2id, self.background_ref_start_arrays[index], self.background_ref_end_arrays[index]]

    def __len__(self):
        return self.len

    def input(self,id):
        return self.contexts[id]

    def output(self,id):
        return self.outputs[id]

    def background(self,id):
        return self.backgrounds[id]

def train_collate_fn(data):
    id,id_a,context,context_a,query,query_a,background,background_a,output,output_a,background_copy_a,bg_dyn_id2vocab,bg_dyn_vocab2id,tgt_vocab2id, background_ref_start_a, background_ref_end_a = zip(*data)

    batch_size=len(id_a)

    id_t = torch.cat(id_a)
    context_t = torch.zeros(batch_size, max([len(s) for s in context_a]), requires_grad=False).long()
    query_t = torch.zeros(batch_size, max([len(s) for s in query_a]), requires_grad=False).long()
    output_t = torch.zeros(batch_size, max([len(s) for s in output_a]), requires_grad=False).long()
    background_copy_t = torch.zeros(batch_size, max([len(s) for s in background_copy_a]), requires_grad=False).long()
    background_t = torch.zeros(batch_size, max([len(s) for s in background_a]), requires_grad=False).long()
    # bg_sel_t = torch.zeros(batch_size, max([len(s) for s in bg_sel_a]), requires_grad=False).long()
    background_ref_start_t = torch.cat(background_ref_start_a)
    background_ref_end_t = torch.cat(background_ref_end_a)

    background_map_t=torch.zeros(batch_size, max([len(s) for s in background_a]), max([len(s) for k,s in bg_dyn_vocab2id]), requires_grad=False).long()

    output_text=dict()
    background_text=dict()

    def one_instance(b):
        context_t[b, :len(context_a[b])] = context_a[b]
        query_t[b, :len(query_a[b])] = query_a[b]
        output_t[b, :len(output_a[b])] = output_a[b]
        background_copy_t[b, :len(background_copy_a[b])] = background_copy_a[b]
        background_t[b, :len(background_a[b])] = background_a[b]
        # bg_sel_t[b, :len(bg_sel_a[b])] = bg_sel_a[b]

        output_text[id_a[b].item()]=output[b]
        background_text[id_a[b].item()]=background[b]

        _, vocab2id = bg_dyn_vocab2id[b]
        for j in range(len(background[b])):
            # if j >= background_ref_start_a[b] and j <= background_ref_end_a[b]:
            background_map_t[b, j, vocab2id[background[b][j].lower()]] = 1

    for b in range(batch_size):
        one_instance(b)

    # pool.map(one_instance, range(batch_size))

    return {'id':id_t,'context':context_t, 'query':query_t, 'output':output_t,'output_text':output_text, 'background':background_t, 'background_text': background_text, 'background_ref_start':background_ref_start_t,'background_ref_end':background_ref_end_t, 'background_dyn_vocab':dict(bg_dyn_id2vocab), 'background_copy':background_copy_t, 'background_map':background_map_t}


def test_collate_fn(data):
    id, id_a, context, context_a,query,query_a, background, background_a, output, output_a, background_copy_a, bg_dyn_id2vocab, bg_dyn_vocab2id, tgt_vocab2id, background_ref_start_a, background_ref_end_a = zip(
        *data)
    tgt_vocab2id = tgt_vocab2id[0]

    batch_size = len(id_a)

    id_t = torch.cat(id_a)
    context_t = torch.zeros(batch_size, max([len(s) for s in context_a]), requires_grad=False).long()
    query_t = torch.zeros(batch_size, max([len(s) for s in query_a]), requires_grad=False).long()
    output_t = torch.zeros(batch_size, max([len(s) for s in output_a]), requires_grad=False).long()
    background_copy_t = torch.zeros(batch_size, max([len(s) for s in background_copy_a]), requires_grad=False).long()
    background_t = torch.zeros(batch_size, max([len(s) for s in background_a]), requires_grad=False).long()
    # bg_sel_t = torch.zeros(batch_size, max([len(s) for s in bg_sel_a]), requires_grad=False).long()

    background_map_t = torch.zeros(batch_size, max([len(s) for s in background_a]), max([len(s) for k, s in bg_dyn_vocab2id]),
                                   requires_grad=False).long()
    background_vocab_map_t = torch.zeros(batch_size, max([len(s) for k, s in bg_dyn_id2vocab]), len(tgt_vocab2id),
                                         requires_grad=False).float()
    background_vocab_overlap_t = torch.ones(batch_size, max([len(s) for k, s in bg_dyn_id2vocab]),
                                            requires_grad=False).float()
    background_text=dict()

    def one_instance(b):
        context_t[b, :len(context_a[b])] = context_a[b]
        query_t[b, :len(query_a[b])] = query_a[b]
        output_t[b, :len(output_a[b])] = output_a[b]
        background_copy_t[b, :len(background_copy_a[b])] = background_copy_a[b]
        background_t[b, :len(background_a[b])] = background_a[b]
        # bg_sel_t[b, :len(bg_sel_a[b])] = bg_sel_a[b]

        background_text[id_a[b].item()]=background[b]

        _, vocab2id = bg_dyn_vocab2id[b]
        for j in range(len(background[b])):
            # if j >= background_ref_start_a[b] and j <= background_ref_end_a[b]:
            background_map_t[b, j, vocab2id[background[b][j].lower()]] = 1

        _, id2vocab = bg_dyn_id2vocab[b]
        for id in id2vocab:
            if id2vocab[id] in tgt_vocab2id:
                background_vocab_map_t[b, id, tgt_vocab2id[id2vocab[id]]] = 1
                background_vocab_overlap_t[b, id] = 0

    for b in range(batch_size):
        one_instance(b)

    # pool.map(one_instance, range(batch_size))

    return {'id': id_t, 'context':context_t, 'query':query_t, 'output':output_t, 'background':background_t,
            'background_dyn_vocab': dict(bg_dyn_id2vocab), 'background_copy': background_copy_t,
            'background_map': background_map_t, 'background_vocab_map': background_vocab_map_t,
            'background_vocab_overlap': background_vocab_overlap_t, 'background_text': background_text}

