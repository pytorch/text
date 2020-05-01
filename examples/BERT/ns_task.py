import argparse
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
from model import NextSentenceTask
from utils import setup, cleanup, run_demo, print_loss_log
import torch.distributed as dist
import os


def generate_next_sentence_data(whole_data, args):
    processed_data = []

    for item in whole_data:
        if len(item) > 1:
            # idx to split the text into two sentencd
            split_idx = torch.randint(1, len(item), size=(1, 1)).item()
            # Index 2 means same sentence label. Initial true int(1)
            processed_data.append([item[:split_idx], item[split_idx:], 1])

    # Random shuffle data to have args.frac_ns next sentence set up
    shuffle_idx1 = torch.randperm(len(processed_data))
    shuffle_idx2 = torch.randperm(len(processed_data))
    num_shuffle = int(len(processed_data) * args.frac_ns)
    shuffle_zip = list(zip(shuffle_idx1, shuffle_idx2))[:num_shuffle]
    for (i, j) in shuffle_zip:
        processed_data[i][1] = processed_data[j][0]
        processed_data[i][2] = int(0)  # Switch same sentence label to false 0
    return processed_data


def pad_next_sentence_data(batch, args, sep_id, pad_id):
    # Fix sequence length to args.bptt with padding or trim
    seq_list = []
    tok_type = []
    same_sentence_labels = []
    for item in batch:
        qa_item = torch.tensor(item[0] + [sep_id] + item[1] + [sep_id])
        if qa_item.size(0) > args.bptt:
            qa_item = qa_item[:args.bptt]
        elif qa_item.size(0) < args.bptt:
            qa_item = torch.cat((qa_item,
                                 torch.tensor([pad_id] * (args.bptt -
                                              qa_item.size(0)))))
        seq_list.append(qa_item)
        _tok_tp = torch.ones((qa_item.size(0)))
        _idx = min(len(item[0]) + 1, args.bptt)
        _tok_tp[:_idx] = 0.0
        tok_type.append(_tok_tp)
        same_sentence_labels.append(item[2])

    return torch.stack(seq_list).long().t().contiguous(), \
        torch.stack(tok_type).long().t().contiguous(), \
        torch.tensor(same_sentence_labels).long().contiguous()


###############################################################################
# Evaluating code
###############################################################################


def evaluate(data_source, model, device, criterion, sep_id, pad_id, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    batch_size = args.batch_size
    dataloader = DataLoader(data_source, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda b: pad_next_sentence_data(b, args, sep_id, pad_id))
    cls_id = data_source.vocab.stoi['<cls>']

    with torch.no_grad():
        for idx, (seq_input, tok_type, target_ns_labels) in enumerate(dataloader):
            # Add <'cls'> token id to the beginning of seq across batches
            seq_input = torch.cat((torch.tensor([[cls_id] * seq_input.size(1)]).long(), seq_input))
            tok_type = torch.cat((torch.tensor([[0] * tok_type.size(1)]).long(), tok_type))
            if args.parallel == 'DDP':
                seq_input = seq_input.to(device[0])
                tok_type = tok_type.to(device[0])
                target_ns_labels = target_ns_labels.to(device[0])
            else:
                seq_input = seq_input.to(device)
                tok_type = tok_type.to(device)
                target_ns_labels = target_ns_labels.to(device)
            seq_input = seq_input.transpose(0, 1)  # Wrap up by DDP or DataParallel
            ns_labels = model(seq_input, token_type_input=tok_type)
            loss = criterion(ns_labels, target_ns_labels)
            total_loss += loss.item()

    return total_loss / (len(data_source) // batch_size)


###############################################################################
# Training code
###############################################################################

def train(train_dataset, model, train_loss_log, device, optimizer, criterion,
          epoch, scheduler, sep_id, pad_id, args, rank=None):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    batch_size = args.batch_size
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda b: pad_next_sentence_data(b, args, sep_id, pad_id))
    cls_id = train_dataset.vocab.stoi['<cls>']
    train_loss_log.append(0.0)

    for idx, (seq_input, tok_type, target_ns_labels) in enumerate(dataloader):
        # Add <'cls'> token id to the beginning of seq across batches
        seq_input = torch.cat((torch.tensor([[cls_id] * seq_input.size(1)]).long(), seq_input))
        tok_type = torch.cat((torch.tensor([[0] * tok_type.size(1)]).long(), tok_type))
        if args.parallel == 'DDP':
            seq_input = seq_input.to(device[0])
            tok_type = tok_type.to(device[0])
            target_ns_labels = target_ns_labels.to(device[0])
        else:
            seq_input = seq_input.to(device)
            tok_type = tok_type.to(device)
            target_ns_labels = target_ns_labels.to(device)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        seq_input = seq_input.transpose(0, 1)  # Wrap up by DDP or DataParallel
        ns_labels = model(seq_input, token_type_input=tok_type)
        loss = criterion(ns_labels, target_ns_labels)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        total_loss += loss.item()

        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            if (rank is None) or rank == 0:
                train_loss_log[-1] = cur_loss
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | '
                      'ms/batch {:5.2f} | '
                      'loss {:8.5f} | ppl {:5.2f}'.format(epoch, idx,
                                                          len(train_dataset) // batch_size,
                                                          scheduler.get_last_lr()[0],
                                                          elapsed * 1000 / args.log_interval,
                                                          cur_loss, math.exp(cur_loss)))
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

    ###################################################################
    # Load data
    ###################################################################
    vocab = torch.load(args.save_vocab)
    pad_id = vocab.stoi['<pad>']
    sep_id = vocab.stoi['<sep>']

    if args.dataset == 'WikiText103':
        from data import WikiText103
        train_dataset, valid_dataset, test_dataset = WikiText103(vocab=vocab,
                                                                 single_line=False)
    elif args.dataset == 'BookCorpus':
        from data import BookCorpus
        train_dataset, valid_dataset, test_dataset = BookCorpus(vocab=vocab,
                                                                min_sentence_len=args.min_sentence_len)

    if rank is not None:
        chunk_len = len(train_dataset.data) // args.world_size
        train_dataset.data = train_dataset.data[(rank * chunk_len):((rank + 1) * chunk_len)]
    train_dataset.data = generate_next_sentence_data(train_dataset.data, args)
    valid_dataset.data = generate_next_sentence_data(valid_dataset.data, args)
    test_dataset.data = generate_next_sentence_data(test_dataset.data, args)

    ###################################################################
    # Build the model
    ###################################################################
    pretrained_bert = torch.load(args.bert_model)
    model = NextSentenceTask(pretrained_bert)
    if args.checkpoint != 'None':
        model = torch.load(args.checkpoint)

    if args.parallel == 'DDP':
        model = model.to(device[0])
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=device)
    else:
        model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    ###################################################################
    # Loop over epochs.
    ###################################################################

    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    best_val_loss = None
    train_loss_log, val_loss_log = [], []

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train(train_dataset, model, train_loss_log, device, optimizer,
              criterion, epoch, scheduler, sep_id, pad_id, args, rank)
        val_loss = evaluate(valid_dataset, model, device, criterion,
                            sep_id, pad_id, args)
        val_loss_log.append(val_loss)

        if (rank is None) or (rank == 0):
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s '
                  '| valid loss {:8.5f} | '.format(epoch,
                                                   (time.time() - epoch_start_time),
                                                   val_loss))
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
    ###################################################################
    # Load the best saved model and run on test data
    ###################################################################
    if args.parallel == 'DDP':
        # [TODO] put dist.barrier() back
        # dist.barrier()
        # configure map_location properly
        rank0_devices = [x - rank * len(device) for x in device]
        device_pairs = zip(rank0_devices, device)
        map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
        model.load_state_dict(torch.load(os.environ['SLURM_JOB_ID'] + '_' + args.save, map_location=map_location))
        test_loss = evaluate(test_dataset, model, device, criterion, sep_id, pad_id, args)
        if rank == 0:
            print('=' * 89)
            print('| End of training | test loss {:8.5f} | test ppl {:8.5f}'.format(
                  test_loss, math.exp(test_loss)))
            print('=' * 89)
            print_loss_log(os.environ['SLURM_JOB_ID'] + '_ns_loss.txt', train_loss_log, val_loss_log, test_loss, args)
        ###############################################################################
        # Save the bert model layer
        ###############################################################################
            with open(os.environ['SLURM_JOB_ID'] + '_' + args.save, 'wb') as f:
                torch.save(model.module.bert_model, f)
            with open(os.environ['SLURM_JOB_ID'] + '_' + 'full_ns_model.pt', 'wb') as f:
                torch.save(model.module, f)
    else:
        with open(args.save, 'rb') as f:
            model = torch.load(f)

        test_loss = evaluate(test_dataset, model, device,
                             criterion, sep_id, pad_id)
        print('=' * 89)
        print('| End of training | test loss {:8.5f} | test ppl {:8.5f}'.format(
              test_loss, math.exp(test_loss)))
        print('=' * 89)
        print_loss_log('ns_loss.txt', train_loss_log, val_loss_log, test_loss)

        with open(args.save, 'wb') as f:
            torch.save(model.bert_model, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Question-Answer fine-tuning task')
    parser.add_argument('--dataset', type=str, default='WikiText103',
                        help='dataset used for next sentence task')
    parser.add_argument('--lr', type=float, default=5,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=2,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=6, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='max. sequence length for the next-sentence pair')
    parser.add_argument('--min_sentence_len', type=int, default=1,
                        help='min. sequence length for the raw text tokens')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='report interval')
    parser.add_argument('--checkpoint', type=str, default='None',
                        help='path to load the checkpoint')
    parser.add_argument('--save', type=str, default='ns_model.pt',
                        help='path to save the final model')
    parser.add_argument('--save-vocab', type=str, default='vocab.pt',
                        help='path to save the vocab')
    parser.add_argument('--bert-model', type=str,
                        help='path to save the pretrained bert')
    parser.add_argument('--frac_ns', type=float, default=0.5,
                        help='fraction of not next sentence')
    parser.add_argument('--parallel', type=str, default='None',
                        help='Use DataParallel/DDP to train model')
    parser.add_argument('--world_size', type=int, default=1,
                        help='the world size to initiate DPP')
    args = parser.parse_args()

    if args.parallel == 'DDP':
        run_demo(run_ddp, args)
    else:
        run_main(args)
