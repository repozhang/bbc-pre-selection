from MALCopyDataset import *
from MAL import *
from torch import optim
from trainers.DefaultTrainer import *
import torch.backends.cudnn as cudnn

if __name__ == '__main__':

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())

    init_seed(123456)

    data_path = 'dataset/holl.256/'

    output_path = 'log_mal/holl.256/'


    src_vocab2id, src_id2vocab, src_id2freq = load_vocab(data_path + 'holl_input_output.256.vocab', t=10)
    tgt_vocab2id, tgt_id2vocab, tgt_id2freq = src_vocab2id, src_id2vocab, src_id2freq

    train_dataset = MALDataset([data_path + 'holl-train.256.json'], src_vocab2id, tgt_vocab2id)
    dev_dataset = MALDataset([data_path + 'holl-dev.256.json'], src_vocab2id, tgt_vocab2id)
    test_dataset = MALDataset([data_path + 'holl-test.256.json'], src_vocab2id, tgt_vocab2id)


    # env = Environment(128, 128, 256)
    encoder=Encoder(len(src_vocab2id), 128, 256)
    selector =Selector(128, 256, len(tgt_vocab2id))
    generator = Generator(128, 256, len(tgt_vocab2id))
    model=MAL(encoder, selector, generator, None, src_id2vocab, src_vocab2id, tgt_id2vocab, tgt_vocab2id, max_dec_len=50, beam_width=1)
    init_params(model)

    # env_optimizer = optim.Adam(filter(lambda x: x.requires_grad, env.parameters()))
    # selector_optimizer = optim.Adam(filter(lambda x: x.requires_grad, selector.parameters()))
    # generator_optimizer = optim.Adam(filter(lambda x: x.requires_grad, generator.parameters()))
    agent_optimizer = optim.Adam(filter(lambda x: x.requires_grad, list(selector.parameters()) + list(generator.parameters())))
    model_optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()))

    batch_size = 8
    # batch_size = 32

    trainer = DefaultTrainer(model)

    for i in range(100):
        trainer.train_epoch('mle_train', train_dataset, train_collate_fn, batch_size, i, model_optimizer)
        rouges = trainer.test(dev_dataset, test_collate_fn, batch_size, i, output_path=output_path)
        rouges = trainer.test(test_dataset, test_collate_fn, batch_size, 100+i, output_path=output_path)
        trainer.serialize(i, output_path=output_path)