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


def collate_batch(batch_data, args, mask_id, tokenizer, vocab):
    output_tensor = []
    language_type = []
    for (language_id, line) in batch_data:
        ids = vocab(tokenizer(line))
        if len(ids) > args.bptt:  # Control the max length of the sequences
            ids = ids[:args.bptt]
        output_tensor += ids
        language_type += [language_id] * len(ids)

    nseq = len(output_tensor) // args.batch_size
    batch_data = torch.tensor(output_tensor[:(args.batch_size * nseq)],
                              dtype=torch.long).view(args.batch_size, -1).t().contiguous()
    language_type = torch.tensor(language_type[:(args.batch_size * nseq)],
                                 dtype=torch.long).view(args.batch_size, -1).t().contiguous()

    # Generate masks with args.mask_frac
    nseq = batch_data.size(0)
    ones_num = max(1, int(nseq * args.mask_frac))  # To ensure targets is not empty
    zeros_num = nseq - ones_num
    lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
    lm_mask = lm_mask[torch.randperm(nseq)]
    targets = torch.stack([batch_data[i] for i in range(lm_mask.size(0)) if lm_mask[i]]).view(-1)
    batch_data = batch_data.masked_fill(lm_mask.bool().unsqueeze(1), mask_id)
    return batch_data, language_type, lm_mask, targets


def train(model, mask_id, train_loss_log, train_data, tokenizer, vocab,
          optimizer, criterion, ntokens, epoch, scheduler, args, device, rank=None):
    model.train()
    total_loss = 0.
    start_time = time.time()
    train_loss_log.append(0.0)
    dataloader = DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=False, collate_fn=lambda b: collate_batch(b, args, mask_id, tokenizer, vocab))

    for batch, (data, language_type, lm_mask, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        data = data.to(device)
        language_type = language_type.to(device)
        targets = targets.to(device)
        output = model(data, language_type)
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
    mask_id = vocab(['<MASK>'])[0]
    ntokens = len(vocab)

    # dataset = CC100('/datasets01/cc100/031720/', {'as_IN.txt', 'om_KE.txt', 'su_ID.txt'}, start_line=300, chunk=300)
    dataset = CC100('/datasets01/cc100/031720/', {'*.txt'}, start_line=200, chunk=500)

    model = CrossLingualMLMTask(ntokens, args.emsize, 115, args.nhead, args.nhid, args.nlayers, args.dropout)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    train_loss_log = []

    for epoch in range(1, args.epochs + 1):
        # epoch_start_time = time.time()
        train(model, mask_id, train_loss_log, dataset, tokenizer, vocab,
              optimizer, criterion, ntokens, epoch, scheduler, args, device, rank)


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
    parser.add_argument('--epochs', type=int, default=3,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=128,
                        help='max. sequence length')
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

    run_main(args)
