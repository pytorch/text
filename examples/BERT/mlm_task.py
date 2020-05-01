import argparse
import time
import math
import torch
import torch.nn as nn
from model import MLMTask
from utils import setup, cleanup, run_demo, print_loss_log
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os


def batchify(txt_data, bsz, args):

    # Cut the data to bptt and bsz
    _num = len(txt_data) // (bsz * args.bptt)
    txt_data = txt_data[:(_num * bsz * args.bptt)]
    # Divide the dataset into bsz parts.
    nbatch = txt_data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    txt_data = txt_data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    txt_data = txt_data.view(bsz, -1).t().contiguous()
    return txt_data


###############################################################################
# Training code
###############################################################################

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, args):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    return data


def evaluate(data_source, model, vocab, ntokens, criterion, args, device):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    mask_id = vocab.stoi['<MASK>']
    cls_id = vocab.stoi['<cls>']
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):

            if i + args.bptt * args.world_size > len(data_source):
                continue
            data = get_batch(data_source, i, args)
            # Generate masks with args.mask_frac
            data_len = data.size(0)
            ones_num = int(data_len * args.mask_frac)
            zeros_num = data_len - ones_num
            lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
            lm_mask = lm_mask[torch.randperm(data_len)]

            # Add <'cls'> token id to the beginning of seq across batches
            data = torch.cat((torch.tensor([[cls_id] * data.size(1)]).long(), data))
            lm_mask = torch.cat((torch.tensor([0.0]), lm_mask))

            targets = torch.stack([data[i] for i in range(lm_mask.size(0)) if lm_mask[i]]).view(-1)
            data = data.masked_fill(lm_mask.bool().unsqueeze(1), mask_id)

            data = data.transpose(0, 1)  # Wrap up by nn.DataParallel
            output = model(data)
            output = torch.stack([output[i] for i in range(lm_mask.size(0)) if lm_mask[i]])
            output_flat = output.view(-1, ntokens)
            total_loss += criterion(output_flat, targets.to(device[0])).item()
    return total_loss / ((len(data_source) - 1) / args.bptt)


def train(model, vocab, train_loss_log, train_data,
          optimizer, criterion, ntokens, epoch, scheduler, args, device, rank=None):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    mask_id = vocab.stoi['<MASK>']
    cls_id = vocab.stoi['<cls>']
    train_loss_log.append(0.0)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):

        data = get_batch(train_data, i, args)

        # Generate masks with args.mask_frac
        data_len = data.size(0)
        ones_num = int(data_len * args.mask_frac)
        zeros_num = data_len - ones_num
        lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
        lm_mask = lm_mask[torch.randperm(data_len)]

        # Add <'cls'> token id to the beginning of seq across batches
        data = torch.cat((torch.tensor([[cls_id] * data.size(1)]).long(), data))
        lm_mask = torch.cat((torch.tensor([0.0]), lm_mask))

        targets = torch.stack([data[i] for i in range(lm_mask.size(0)) if lm_mask[i]]).view(-1)
        data = data.masked_fill(lm_mask.bool().unsqueeze(1), mask_id)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        data = data.transpose(0, 1)  # Wrap up by nn.DataParallel
        output = model(data)
        output = torch.stack([output[i] for i in range(lm_mask.size(0)) if lm_mask[i]])
        loss = criterion(output.view(-1, ntokens), targets.to(device[0]))
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            if (rank is None) or rank == 0:
                train_loss_log[-1] = cur_loss
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(train_data) // args.bptt, scheduler.get_last_lr()[0],
                      elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def run_ddp(rank, args):
    setup(rank, args.world_size, args.seed)
    run_main(args, rank)
    cleanup()


def run_main(args, rank=None):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if args.parallel == 'DDP':
        n = torch.cuda.device_count() // args.world_size
        device = list(range(rank * n, (rank + 1) * n))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###############################################################################
    # Import dataset
    ###############################################################################
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

    try:
        vocab = torch.load(args.save_vocab)
    except:
        train_dataset, test_dataset, valid_dataset = WLMDataset()
        old_vocab = train_dataset.vocab
        vocab = torchtext.vocab.Vocab(counter=old_vocab.freqs,
                                      specials=['<unk>', '<pad>', '<MASK>'])
        with open(args.save_vocab, 'wb') as f:
            torch.save(vocab, f)

    if args.dataset == 'WikiText103' or args.dataset == 'WikiText2':
        train_dataset, test_dataset, valid_dataset = WLMDataset(vocab=vocab)
    elif args.dataset == 'WMTNewsCrawl':
        test_dataset, valid_dataset = torchtext.experimental.datasets.WikiText2(vocab=vocab, data_select=('test', 'valid'))
        train_dataset, = WLMDataset(vocab=vocab, data_select='train')
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
        train_dataset, test_dataset, valid_dataset = BookCorpus(vocab)

    train_data = batchify(train_dataset.data, args.batch_size, args)

    if rank is not None:
        # Chunk training data by rank for different gpus
        chunk_len = len(train_data) // args.world_size
        train_data = train_data[(rank*chunk_len):((rank + 1)*chunk_len)]

    val_data = batchify(valid_dataset.data, args.batch_size, args)
    test_data = batchify(test_dataset.data, args.batch_size, args)

    ###############################################################################
    # Build the model
    ###############################################################################
    ntokens = len(train_dataset.get_vocab())
    model = MLMTask(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout)
    if args.checkpoint != 'None':
        model.bert_model = torch.load(args.checkpoint)

    if args.parallel == 'DDP':
        model = model.to(device[0])
        # model = nn.DataParallel(model)  # Wrap up by nn.DataParallel
        model = DDP(model, device_ids=device)
    else:
        model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    ###############################################################################
    # Loop over epochs.
    ###############################################################################
    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    best_val_loss = None
    train_loss_log, val_loss_log = [], []

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train(model, train_dataset.vocab, train_loss_log, train_data,
              optimizer, criterion, ntokens, epoch, scheduler, args, device, rank)
        # train()
        val_loss = evaluate(val_data, model, train_dataset.vocab, ntokens, criterion, args, device)

        if (rank is None) or (rank == 0):
            val_loss_log.append(val_loss)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            if rank is None:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
            elif rank == 0:
                with open(os.environ['SLURM_JOB_ID'] + '_' + args.save, 'wb') as f:
                    torch.save(model.state_dict(), f)
            best_val_loss = val_loss
        else:
            scheduler.step()

    ###############################################################################
    # Load the best saved model.
    ###############################################################################
    if args.parallel == 'DDP':
        dist.barrier()
        # configure map_location properly
        rank0_devices = [x - rank * len(device) for x in device]
        device_pairs = zip(rank0_devices, device)
        map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
        model.load_state_dict(
            torch.load(os.environ['SLURM_JOB_ID'] + '_' + args.save, map_location=map_location))

        ###############################################################################
        # Run on test data.
        ###############################################################################
        test_loss = evaluate(test_data, model, train_dataset.vocab, ntokens, criterion, args, device)
        if rank == 0:
            print('=' * 89)
            print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
                  test_loss, math.exp(test_loss)))
            print('=' * 89)
            print_loss_log(os.environ['SLURM_JOB_ID'] + '_mlm_loss.txt', train_loss_log, val_loss_log, test_loss, args)

            ###############################################################################
            # Save the bert model layer
            ###############################################################################
            with open(os.environ['SLURM_JOB_ID'] + '_' + args.save, 'wb') as f:
                torch.save(model.module.bert_model, f)
            with open(os.environ['SLURM_JOB_ID'] + '_mlm_model.pt', 'wb') as f:
                torch.save(model.module, f)
    else:
        with open(args.save, 'rb') as f:
            model = torch.load(f)
        test_loss = evaluate(test_data, model, train_dataset.vocab, ntokens, criterion, args, device)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
              test_loss, math.exp(test_loss)))
        print('=' * 89)
        print_loss_log('mlm_loss.txt', train_loss_log, val_loss_log, test_loss, args)

        ###############################################################################
        # Save the bert model layer
        ###############################################################################
        with open(args.save, 'wb') as f:
            torch.save(model.module.bert_model, f)
        with open('mlm_model.pt', 'wb') as f:
            torch.save(model.module, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Transformer Language Model')
    parser.add_argument('--emsize', type=int, default=32,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=64,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=4,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=8,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--lr', type=float, default=5,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=2,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='report interval')
    parser.add_argument('--checkpoint', type=str, default='None',
                        help='path to load the checkpoint')
    parser.add_argument('--save', type=str, default='bert_model.pt',
                        help='path to save the final model')
    parser.add_argument('--save-vocab', type=str, default='vocab.pt',
                        help='path to save the vocab')
    parser.add_argument('--mask_frac', type=float, default=0.15,
                        help='the fraction of masked tokens')
    parser.add_argument('--dataset', type=str, default='WikiText2',
                        help='dataset used for MLM task')
    parser.add_argument('--parallel', type=str, default='None',
                        help='Use DataParallel to train model')
    parser.add_argument('--world_size', type=int, default=1,
                        help='the world size to initiate DPP')
    args = parser.parse_args()

    if args.parallel == 'DDP':
        run_demo(run_ddp, args)
    else:
        run_main(args)
