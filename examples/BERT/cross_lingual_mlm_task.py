import argparse
import time
import math
import torch
import torch.nn as nn
from data import CC100
from model import CrossLingualMLMTask
from torch.utils.data import DataLoader
from torchtext.experimental.transforms import sentencepiece_tokenizer
from transforms import PretrainedSPVocab
from torch.nn.utils.rnn import pad_sequence


def collate_batch(batch_data, args, mask_id, pad_id, text_transform):
    output_tensor = []
    mask_tensor = []
    for (language_id, line) in batch_data:
        # ids = vocab(tokenizer(line))
        ids = text_transform(line)
        if len(ids) > args.bptt:  # Control the max length of the sequences
            ids = ids[:args.bptt]
        output_tensor.append(torch.tensor(ids, dtype=torch.long))
        nseq = len(ids)
        ones_num = max(1, int(nseq * args.mask_frac))  # To ensure targets is not empty
        zeros_num = nseq - ones_num
        lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
        lm_mask = lm_mask[torch.randperm(nseq)]
        mask_tensor.append(lm_mask)

    output_tensor = pad_sequence(output_tensor, padding_value=pad_id)
    mask_tensor = pad_sequence(mask_tensor, padding_value=0.0)
    batch_data = output_tensor.masked_fill(mask_tensor.bool(), mask_id).to(torch.long)
    targets = output_tensor.masked_fill(mask_tensor.bool() != True, pad_id).to(torch.long)
    return batch_data, targets


def evaluate(data_source, model, mask_id, pad_id, ntokens, criterion, args, device, text_transform):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    dataloader = DataLoader(data_source, batch_size=args.batch_size,
                            shuffle=False, collate_fn=lambda b: collate_batch(b, args, mask_id, pad_id, text_transform))
    with torch.no_grad():
        for batch, (data, targets) in enumerate(dataloader):
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            total_loss += criterion(output.view(-1, ntokens), targets.view(-1)).item()
    return total_loss / ((len(data_source) - 1) / args.batch_size)


def train(model, mask_id, pad_id, train_loss_log, train_data, text_transform,
          optimizer, criterion, ntokens, epoch, scheduler, args, device, rank=None):
    model.train()
    total_loss = 0.
    start_time = time.time()
    train_loss_log.append(0.0)
    dataloader = DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=False, collate_fn=lambda b: collate_batch(b, args, mask_id, pad_id, text_transform))

    for batch, (data, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        data = data.to(device)
        targets = targets.to(device)
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets.view(-1))
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
                                                          len(train_data) // args.batch_size,
                                                          scheduler.get_last_lr()[0],
                                                          elapsed * 1000 / args.log_interval,
                                                          cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def run_main(args, rank=None):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up tokenizer and vocab
    tokenizer = sentencepiece_tokenizer(args.spm_path)
    vocab = PretrainedSPVocab(args.spm_path)
    text_transform = lambda x: vocab(tokenizer(x))
    mask_id = vocab(['<MASK>'])[0]
    pad_id = vocab(['pad'])[0]
    ntokens = len(vocab)

    model = CrossLingualMLMTask(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.75)
    best_val_loss = None
    train_loss_log, val_loss_log = [], []

    for epoch in range(1, args.epochs + 1):
        train_data = CC100('/datasets01/cc100/031720/', {'*.txt'}, start_line=args.start_line, num_lines=args.num_lines)
        # train_data = CC100('/datasets01/cc100/031720/', {'*.txt'}, start_line=200, chunk=5)
        from torchtext.experimental.datasets.raw import WikiText2
        val_data, = WikiText2(data_select='valid')
        val_data = [(17, item) for item in val_data if item != ' \n']  # english language type is 17 in CC100 dataset

        epoch_start_time = time.time()
        train(model, mask_id, pad_id, train_loss_log, train_data, text_transform,
              optimizer, criterion, ntokens, epoch, scheduler, args, device, rank)

        val_loss = evaluate(val_data, model, mask_id, pad_id, ntokens, criterion, args, device, text_transform)
        val_loss_log.append(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            scheduler.step()

    # Run reference XLM-R model from fairseq
    if args.eval_ref != 'None':
        from fairseq.models.roberta import XLMRModel
        xlmr_model = XLMRModel.from_pretrained('./xlmr.large', checkpoint_file='model.pt')
        xlmr_model.eval()
        text_transform = xlmr_model.encode()
        val_loss = evaluate(val_data, xlmr_model, mask_id, pad_id, ntokens, criterion, args, device, text_transform)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Cross-lingual XLM MLM')
    parser.add_argument('--emsize', type=int, default=768,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=3072,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=12,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=12,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.1,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=6,
                        help='upper epoch limit')
    parser.add_argument('--start_line', type=int, default=0,
                        help='the starting line to read text in each file')
    parser.add_argument('--num_lines', type=int, default=50,
                        help='the number of lines to read')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=128,
                        help='max. sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=5431916812,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='report interval')
    parser.add_argument('--save', type=str, default='cross_lingual_mlm_bert.pt',
                        help='path to save the final model')
    parser.add_argument('--spm-path', type=str, default='./sentencepiece.xlmr.model',
                        help='path to load the sentencepiece model')
    parser.add_argument('--eval_ref', type=str, default='None',
                        help='path to load the reference model for evaluation')
    parser.add_argument('--mask_frac', type=float, default=0.15,
                        help='the fraction of masked tokens')
    args = parser.parse_args()

    run_main(args)
