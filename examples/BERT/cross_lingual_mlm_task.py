import argparse
import math
import os
import time
from typing import List

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from data import CC100
from model import CrossLingualMLMTask
from pipeline import SingleProcessPipeline, RPCPipeline, RemoteBaseCPURPC, RemoteBaseCUDARPC
from shard_model import XLMRModelShards, MLMShards
from torchtext.experimental.transforms import sentencepiece_tokenizer
from transforms import PretrainedSPVocab
from torchtext.experimental.models.utils import count_model_param


def collate_batch(batch_data, args, mask_id, pad_id, text_transform):
    output_tensor = []
    mask_tensor = []
    for (language_id, line) in batch_data:
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


def evaluate(data_source, model, mask_id, pad_id, ntokens, criterion, args, devices, text_transform):
    total_loss = 0.
    dataloader = DataLoader(data_source, batch_size=1,  # Set batch # to 1 for inference
                            shuffle=False, collate_fn=lambda b: collate_batch(b, args, mask_id, pad_id, text_transform))
    with torch.no_grad():
        for batch, (data, targets) in enumerate(dataloader):
            data = data.to(devices[0])
            targets = targets.to(devices[-1])
            output = model(data)
            total_loss += criterion(output.view(-1, ntokens), targets.view(-1)).item()
    return total_loss / (len(data_source) - 1)  # Set batch # to 1 for inference


def local_step(model, data, targets, criterion, optimizer, ntokens):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output.view(-1, ntokens), targets.view(-1))
    loss.backward()
    res = loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()
    return res


def dist_step(model, data, targets, criterion, optimizer, ntokens):
    with dist_autograd.context() as context_id:
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets.view(-1))
        dist_autograd.backward(context_id, [loss])
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step(context_id)
        return loss.item()


def train(model, mask_id, pad_id, train_loss_log, train_data, text_transform,
          optimizer, criterion, ntokens, epoch, last_lr, args, devices, step_impl):
    model.train()
    total_loss = 0.
    start_time = time.time()
    train_loss_log.append(0.0)
    dataloader = DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=False, collate_fn=lambda b: collate_batch(b, args, mask_id, pad_id, text_transform))

    for batch, (data, targets) in enumerate(dataloader):
        data = data.to(devices[0])
        targets = targets.to(devices[-1])
        loss = step_impl(model, data, targets, criterion, optimizer, ntokens)

        total_loss += loss
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            train_loss_log[-1] = cur_loss
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch,
                                                        len(train_data) // args.batch_size,
                                                        last_lr,
                                                        elapsed * 1000 / args.log_interval,
                                                        cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def run_main(args):
    torch.manual_seed(args.seed)

    # Set up tokenizer and vocab
    tokenizer = sentencepiece_tokenizer(args.spm_path)
    vocab = PretrainedSPVocab(args.spm_path)

    def text_transform(x: str) -> List:
        return vocab(tokenizer(x))
    mask_id = vocab(['<MASK>'])[0]
    pad_id = vocab(['pad'])[0]
    ntokens = len(vocab)

    xlmr = XLMRModelShards(ntokens, args.emsize, args.nhead, args.nhid, args.dropout)
    mlm = MLMShards(ntokens, args.emsize)
    devices = [f"cuda:{i}" for i in range(args.gpus)] if torch.cuda.is_available() else ["cpu"]

    if len(devices) == 1:
        # In case of one device combine all layers into a single nn.Sequential
        shards = [nn.Sequential(
            xlmr.xlmr_embed(),
            xlmr.encoder_layers(args.nlayers),
            mlm.mlm()
        )]
    elif len(devices) == 2:
        # In case of two devices split the model right in the middle and
        # put the embeddings and half of encoders to the first shard and
        # another half of encoders and mlm head to the second.
        assert args.nlayers % 2 == 0
        shards = [
            nn.Sequential(
                xlmr.xlmr_embed(),
                xlmr.encoder_layers(args.nlayers // 2)
            ),
            nn.Sequential(
                xlmr.encoder_layers(args.nlayers // 2),
                mlm.mlm()
            )
        ]
    else:
        # In case of more that 2 devices put the embeddings and mlm head
        # to the first and the last shard and split the encoders to equal
        # parts among the rest of the shards
        encoder_gpus = (args.gpus - 2)
        assert args.nlayers % encoder_gpus == 0
        encoders_per_gpu = args.nlayers // encoder_gpus
        shards = [
            xlmr.xlmr_embed(),
            *[xlmr.encoder_layers(encoders_per_gpu) for _ in range(encoder_gpus)],
            mlm.mlm()
        ]

    print('Shards parameters:')
    total = 0
    for i, shard in enumerate(shards):
        params = count_model_param(shard)
        total += params
        print(f'shard{i} = {int(params)}M')
    print(f'total = {int(total)}M')

    print("Allocating memory")
    if args.pipeline_mode == 'sp':
        model = SingleProcessPipeline(shards, devices)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.75)
    else:
        workers = [f"worker{i+1}" for i in range(len(devices))]
        model = RPCPipeline(shards, devices, workers, split_size=args.split_size, remote_base_class=(RemoteBaseCUDARPC if args.pipeline_mode == 'cuda' else RemoteBaseCPURPC))
        optimizer = DistributedOptimizer(
            optim.Adam,
            model.parameter_rrefs(),
            lr=args.lr,
        )
        scheduler = None

    print("Memory allocated")
    # input("Memory allocated, check nvidia-smi for memory consumption")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    best_val_loss = None
    train_loss_log, val_loss_log = [], []

    for epoch in range(1, args.epochs + 1):
        train_data = CC100(args.cc100_path, {'*.txt'}, start_line=args.start_line, num_lines=args.num_lines)
        from torchtext.datasets import WikiText2
        val_data = WikiText2(split='valid')
        val_data = [(17, item) for item in val_data if item != ' \n']  # english language type is 17 in CC100 dataset

        epoch_start_time = time.time()
        last_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
        train(model, mask_id, pad_id, train_loss_log, train_data, text_transform,
              optimizer, criterion, ntokens, epoch, last_lr, args,
              devices if args.pipeline_mode == 'sp' or args.pipeline_mode == 'cuda' else ["cpu"],
              local_step if args.pipeline_mode == 'sp' else dist_step)

        # Turn on evaluation mode which disables dropout.
        model.eval()
        val_loss = evaluate(val_data, model, mask_id, pad_id, ntokens, criterion, args,
                            devices if args.pipeline_mode == 'sp' or args.pipeline_mode == 'cuda' else ["cpu"],
                            text_transform)
        val_loss_log.append(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        if args.pipeline_mode == 'sp' and not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            if scheduler is not None:
                scheduler.step()

    # Run reference XLM-R model from fairseq
    if args.eval_ref != 'None':
        from fairseq.models.roberta import XLMRModel
        ref_model = XLMRModel.from_pretrained(args.eval_ref, checkpoint_file='model.pt')
        ref_model.eval()

        def text_transform(x: str) -> List:
            return ref_model.encode(x).tolist()
        model = ref_model.model.encoder
        model = model.to(devices[0])
        # Turn on evaluation mode which disables dropout.
        model.eval()
        # from fairseq XLM-R model
        # <mask> is attached to the end of the dictionary at the index 250001
        ref_ntokens, mask_id, pad_id = 250002, 250001, 1
        criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

        # fairseq XLM-R requires batch-first input sequence
        def model_forward(nn_model):
            def _forward(x):
                return nn_model(x.transpose(0, 1))[0].transpose(0, 1)
            return _forward
        val_loss = evaluate(val_data, model_forward(model), mask_id, pad_id, ref_ntokens,
                            criterion, args, devices[0], text_transform)
        print('-' * 89)
        print('| reference model | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(val_loss, math.exp(val_loss)))
        print('-' * 89)


def run_worker(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256)

    if rank == 0:
        if args.pipeline_mode == 'cuda':
            for i in range(args.gpus):
                options.set_device_map("worker" + str(i + 1), {i:i})
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=args.gpus+1,
            rpc_backend_options=options
        )
        run_main(args)
    else:
        if args.pipeline_mode == 'cuda':
            if rank == 1:
                options.set_device_map("master", {0:0})
            else:
                options.set_device_map("worker" + str(rank - 1), {(rank - 1):(rank - 2)})
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=args.gpus+1,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()


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
    parser.add_argument('--cc100_path', type=str, default='/datasets01/cc100/031720/',
                        help='path to cc100')
    parser.add_argument('--gpus', type=int, default=1,
                        help='number of GPUs to use')
    parser.add_argument('--pipeline_mode', type=str, default='sp',
                        help='pipeline mode, `cpu` for CPU RPC, `cuda` for CUDA RPC, `sp` for single process pipeline')
    parser.add_argument('--split_size', type=int, default=8,
                        help='split the input batch into micro-batches')
    args = parser.parse_args()

    if args.pipeline_mode == 'sp':
        run_main(args)
    else:
        mp.spawn(run_worker, args=(args,), nprocs=args.gpus+1, join=True)
