import torch.nn as nn
from torch.utils.data import DataLoader
import time
from torch.nn.init import *
from torch.optim.lr_scheduler import *
from modules.Generations import *
import codecs
import os
import sys
from Constants import *
from Rouge import *
import json


def init_params(model):
    for name, param in model.named_parameters():
        print(name, param.size())
        if param.data.dim() > 1:
            xavier_uniform_(param.data)

class DefaultTrainer(object):
    def __init__(self, model):
        super(DefaultTrainer, self).__init__()

        if torch.cuda.is_available():
            self.model =model.cuda()
        else:
            self.model = model
        self.eval_model = self.model

        self.distributed = False
        if torch.cuda.device_count()>1:
            self.distributed=True
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # print('GPU', torch.cuda.current_device(), 'ready')
            torch.distributed.init_process_group(backend='nccl')
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)

    def train_batch(self, epoch, data, method, optimizer):
        optimizer.zero_grad()
        loss = self.model(data, method=method)

        if isinstance(loss, tuple):
            closs = [l.mean().cpu().item() for l in loss]
            # loss = torch.cat([l.mean().view(1) for l in loss]).sum()
            loss = torch.cat(loss, dim=-1).mean()
        else:
            loss = loss.mean()
            closs = [loss.cpu().item()]

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        optimizer.step()
        return closs

    def serialize(self,epoch, output_path):
        output_path = os.path.join(output_path, 'model/')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        torch.save(self.eval_model.state_dict(), os.path.join(output_path, '.'.join([str(epoch), 'pkl'])))

    def train_epoch(self, method, train_dataset, train_collate_fn, batch_size, epoch, optimizer):
        self.model.train()

        train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=batch_size,
                                                   shuffle=True, pin_memory=True)

        start_time = time.time()
        count_batch=0
        for j, data in enumerate(train_loader, 0):
            if torch.cuda.is_available():
                data_cuda = dict()
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data_cuda[key] = value.cuda()
                    else:
                        data_cuda[key] = value
                data = data_cuda
            count_batch += 1

            bloss = self.train_batch(epoch, data, method=method, optimizer=optimizer)

            if j >= 0 and j%100==0:
                elapsed_time = time.time() - start_time
                print('Method', method, 'Epoch', epoch, 'Batch ', count_batch, 'Loss ', bloss, 'Time ', elapsed_time)
                sys.stdout.flush()
            del bloss

        # elapsed_time = time.time() - start_time
        # print(method + ' ', epoch, 'time ', elapsed_time)
        sys.stdout.flush()

    def predict(self,dataset, collate_fn, batch_size, epoch, output_path):
        self.eval_model.eval()

        with torch.no_grad():
            test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                     pin_memory=True, collate_fn=collate_fn,
                                     num_workers=0)

            srcs = []
            systems = []
            references = []
            for k, data in enumerate(test_loader, 0):
                if torch.cuda.is_available():
                    data_cuda=dict()
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            data_cuda[key]=value.cuda()
                        else:
                            data_cuda[key] = value
                    data = data_cuda

                indices = self.eval_model(data, method='test')
                sents=self.eval_model.to_sentence(data,indices)

                srcs += [' '.join(dataset.input(id.item())) for id in data['id']]
                systems += [' '.join(s).replace(LINESEP_WORD, os.linesep).lower() for s in sents]
                for id in data['id']:
                    refs=' '.join(dataset.output(id.item()))
                    references.append(refs.lower())

            output_path = os.path.join(output_path, 'result/')
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            file = codecs.open(os.path.join(output_path, str(epoch)+'.txt'), "w", "utf-8")
            for i in range(len(systems)):
                file.write(systems[i]+ os.linesep)
            file.close()
        return systems,references,data

    def test(self,dataset, collate_fn, batch_size, epoch, output_path):
        with torch.no_grad():
            systems,references,data=self.predict(dataset, collate_fn, batch_size, epoch, output_path)

        rouges= rouge(systems, references)
        scores=rouges
        # bleu=sentence_bleu(systems, references)
        # scores['sentence_bleu']=bleu['sentence_bleu']
        print(scores)
        scores['score']=scores['rouge_l/f_score']

        return scores,data