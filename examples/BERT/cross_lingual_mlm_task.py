import argparse
import time
import math
import torch
import torch.nn as nn
from model import MLMTask
from utils import run_demo, run_ddp, wrap_up
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchtext.experimental.transforms import basic_english_normalize
from torchtext.experimental.transforms import sentencepiece_tokenizer
from transforms import PretrainedSPVocab


def collate_batch(batch_data, args, mask_id, cls_id):
    batch_data = torch.tensor(batch_data).long().view(args.batch_size, -1).t().contiguous()
    # Generate masks with args.mask_frac
    data_len = batch_data.size(0)
    ones_num = int(data_len * args.mask_frac)
    zeros_num = data_len - ones_num
    lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
    lm_mask = lm_mask[torch.randperm(data_len)]
    batch_data = torch.cat((torch.tensor([[cls_id] * batch_data.size(1)]).long(), batch_data))
    lm_mask = torch.cat((torch.tensor([0.0]), lm_mask))

    targets = torch.stack([batch_data[i] for i in range(lm_mask.size(0)) if lm_mask[i]]).view(-1)
    batch_data = batch_data.masked_fill(lm_mask.bool().unsqueeze(1), mask_id)
    return batch_data, lm_mask, targets


def process_raw_data(raw_data, args):
    _num = raw_data.size(0) // (args.batch_size * args.bptt)
    raw_data = raw_data[:(_num * args.batch_size * args.bptt)]
    return raw_data


def evaluate(data_source, model, special_token_id, ntokens, criterion, args, device):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    mask_id, cls_id = special_token_id
    dataloader = DataLoader(data_source, batch_size=args.batch_size * args.bptt,
                            shuffle=False, collate_fn=lambda b: collate_batch(b, args, mask_id, cls_id))
    with torch.no_grad():
        for batch, (data, lm_mask, targets) in enumerate(dataloader):
            if args.parallel == 'DDP':
                data = data.to(device[0])
                targets = targets.to(device[0])
            else:
                data = data.to(device)
                targets = targets.to(device)
            data = data.transpose(0, 1)  # Wrap up by DDP or DataParallel
            output = model(data)
            output = torch.stack([output[i] for i in range(lm_mask.size(0)) if lm_mask[i]])
            output_flat = output.view(-1, ntokens)
            total_loss += criterion(output_flat, targets).item()
    return total_loss / ((len(data_source) - 1) / args.bptt / args.batch_size)


def train(model, special_token_id, train_loss_log, train_data,
          optimizer, criterion, ntokens, epoch, scheduler, args, device, rank=None):
    model.train()
    total_loss = 0.
    start_time = time.time()
    mask_id, cls_id = special_token_id
    train_loss_log.append(0.0)
    dataloader = DataLoader(train_data, batch_size=args.batch_size * args.bptt,
                            shuffle=False, collate_fn=lambda b: collate_batch(b, args, mask_id, cls_id))

    for batch, (data, lm_mask, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        if args.parallel == 'DDP':
            data = data.to(device[0])
            targets = targets.to(device[0])
        else:
            data = data.to(device)
            targets = targets.to(device)
        data = data.transpose(0, 1)  # Wrap up by DDP or DataParallel
        output = model(data)
        output = torch.stack([output[i] for i in range(lm_mask.size(0)) if lm_mask[i]])
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            if (rank is None) or rank == 0:
                train_loss_log[-1] = cur_loss
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch,
                                                          len(train_data) // (args.bptt * args.batch_size),
                                                          scheduler.get_last_lr()[0],
                                                          elapsed * 1000 / args.log_interval,
                                                          cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def run_main(args, rank=None):
    torch.manual_seed(args.seed)
    if args.parallel == 'DDP':
        n = torch.cuda.device_count() // args.world_size
        device = list(range(rank * n, (rank + 1) * n))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import torchtext
    if args.dataset == 'WikiText103':
        from torchtext.experimental.datasets import WikiText103 as WLMDataset
    elif args.dataset == 'WikiText2':
        from torchtext.experimental.datasets import WikiText2 as WLMDataset
    elif args.dataset == 'WMTNewsCrawl':
        from data import WMTNewsCrawl as WLMDataset
    elif args.dataset == 'EnWik9':
        from torchtext.datasets import EnWik9
    elif args.dataset == 'BookCorpus':
        from data import BookCorpus
    else:
        print("dataset for MLM task is not supported")

    # Set up tokenizer and vocab
    if args.spm_path != 'None':
        tokenizer = sentencepiece_tokenizer(args.spm_path)
        vocab = PretrainedSPVocab(args.spm_path)
        special_token_id = vocab(['<MASK>', '<cls>'])
    elif args.save_vocab != 'None':
        tokenizer = basic_english_normalize()
        vocab = torch.load(args.save_vocab)
        special_token_id = (vocab.stoi['<MASK>'], vocab.stoi['<cls>'])
    else:
        tokenizer = basic_english_normalize()
        train_dataset, valid_dataset, test_dataset = WLMDataset()
        old_vocab = train_dataset.vocab
        vocab = torchtext.vocab.Vocab(counter=old_vocab.freqs,
                                      specials=['<unk>', '<pad>', '<MASK>'])
        with open(args.save_vocab, 'wb') as f:
            torch.save(vocab, f)
        special_token_id = (vocab.stoi['<MASK>'], vocab.stoi['<cls>'])
    ntokens = len(vocab)

    if args.dataset == 'WikiText103' or args.dataset == 'WikiText2':
        train_dataset, valid_dataset, test_dataset = WLMDataset(tokenizer=tokenizer, vocab=vocab)
        train_dataset.data = torch.cat(tuple(filter(lambda t: t.numel() > 0, train_dataset)))
        valid_dataset.data = torch.cat(tuple(filter(lambda t: t.numel() > 0, valid_dataset)))
        test_dataset.data = torch.cat(tuple(filter(lambda t: t.numel() > 0, test_dataset)))
    elif args.dataset == 'WMTNewsCrawl':
        from torchtext.experimental.datasets import WikiText2
        test_dataset, valid_dataset = WikiText2(tokenizer=tokenizer, vocab=vocab, data_select=('test', 'valid'))
        valid_dataset.data = torch.cat(tuple(filter(lambda t: t.numel() > 0, valid_dataset)))
        test_dataset.data = torch.cat(tuple(filter(lambda t: t.numel() > 0, test_dataset)))
        train_dataset, = WLMDataset(vocab=vocab, data_select='train')
        train_dataset.data = torch.cat(tuple(filter(lambda t: t.numel() > 0, train_dataset)))
    elif args.dataset == 'EnWik9':
        enwik9 = EnWik9()
        idx1, idx2 = int(len(enwik9) * 0.8), int(len(enwik9) * 0.9)
        train_data = torch.tensor([vocab.stoi[_id]
                                  for _id in enwik9[0:idx1]]).long()
        val_data = torch.tensor([vocab.stoi[_id]
                                 for _id in enwik9[idx1:idx2]]).long()
        test_data = torch.tensor([vocab.stoi[_id]
                                 for _id in enwik9[idx2:]]).long()
        from torchtext.experimental.datasets import LanguageModelingDataset
        train_dataset = LanguageModelingDataset(train_data, vocab)
        valid_dataset = LanguageModelingDataset(val_data, vocab)
        test_dataset = LanguageModelingDataset(test_data, vocab)
    elif args.dataset == 'BookCorpus':
        train_dataset, valid_dataset, test_dataset = BookCorpus(vocab, tokenizer=tokenizer)

    train_data = process_raw_data(train_dataset.data, args)
    if rank is not None:
        # Chunk training data by rank for different gpus
        chunk_len = len(train_data) // args.world_size
        train_data = train_data[(rank * chunk_len):((rank + 1) * chunk_len)]
    val_data = process_raw_data(valid_dataset.data, args)
    test_data = process_raw_data(test_dataset.data, args)

    if args.checkpoint != 'None':
        model = torch.load(args.checkpoint)
    else:
        model = MLMTask(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout)
    if args.parallel == 'DDP':
        model = model.to(device[0])
        model = DDP(model, device_ids=device)
    else:
        model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    best_val_loss = None
    train_loss_log, val_loss_log = [], []

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train(model, special_token_id, train_loss_log, train_data,
              optimizer, criterion, ntokens, epoch, scheduler, args, device, rank)
        val_loss = evaluate(val_data, model, special_token_id, ntokens, criterion, args, device)
        if (rank is None) or (rank == 0):
            val_loss_log.append(val_loss)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            if rank is None:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
            elif rank == 0:
                with open(args.save, 'wb') as f:
                    torch.save(model.state_dict(), f)
            best_val_loss = val_loss
        else:
            scheduler.step()
    if args.parallel == 'DDP':
        dist.barrier()
        rank0_devices = [x - rank * len(device) for x in device]
        device_pairs = zip(rank0_devices, device)
        map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
        model.load_state_dict(
            torch.load(args.save, map_location=map_location))
        test_loss = evaluate(test_data, model, special_token_id, ntokens, criterion, args, device)
        if rank == 0:
            wrap_up(train_loss_log, val_loss_log, test_loss, args, model.module, 'mlm_loss.txt', 'full_mlm_model.pt')
    else:
        with open(args.save, 'rb') as f:
            model = torch.load(f)
        test_loss = evaluate(test_data, model, special_token_id, ntokens, criterion, args, device)
        wrap_up(train_loss_log, val_loss_log, test_loss, args, model, 'mlm_loss.txt', 'full_mlm_model.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Transformer Language Model')
    parser.add_argument('--emsize', type=int, default=768,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=3072,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=12,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=12,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--lr', type=float, default=6,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.1,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=128,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=5431916812,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='report interval')
    parser.add_argument('--checkpoint', type=str, default='None',
                        help='path to load the checkpoint')
    parser.add_argument('--save', type=str, default='mlm_bert.pt',
                        help='path to save the final model')
    parser.add_argument('--save-vocab', type=str, default='torchtext_bert_vocab.pt',
                        help='path to save the vocab')
    parser.add_argument('--spm-path', type=str, default='./sentencepiece.xlmr.model',
                        help='path to load the sentencepiece model')
    parser.add_argument('--mask_frac', type=float, default=0.15,
                        help='the fraction of masked tokens')
    parser.add_argument('--dataset', type=str, default='WikiText2',
                        help='dataset used for MLM task')
    parser.add_argument('--parallel', type=str, default='None',
                        help='Use DataParallel to train model')
    parser.add_argument('--world_size', type=int, default=8,
                        help='the world size to initiate DPP')
    args = parser.parse_args()

#    if args.parallel == 'DDP':
#        run_demo(run_ddp, run_main, args)
#    else:
#        run_main(args)
    from data import CC100
    dataset = CC100('/datasets01/cc100/031720/', {'as_IN.txt', 'om_KE.txt', 'su_ID.txt'}, start_line=300, chunk=10)
    tokenizer = sentencepiece_tokenizer(args.spm_path)
    vocab = PretrainedSPVocab(args.spm_path)
    mask_id = vocab(['<MASK>'])

    def collate_batch(batch):
        output_tensor = []
        for line in batch:
            output_tensor += vocab(tokenizer(line))
        return torch.tensor(output_tensor)
    dataloader = DataLoader(dataset, batch_size=10,
                            shuffle=False, collate_fn=collate_batch)
    for item in dataloader:
        print(item.size(), item)
