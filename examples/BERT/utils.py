import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp
import math


def setup(rank, world_size, seed):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(seed)


def cleanup():
    dist.destroy_process_group()


def run_demo(demo_fn, main_fn, args):
    mp.spawn(demo_fn,
             args=(main_fn, args,),
             nprocs=args.world_size,
             join=True)


def run_ddp(rank, main_fn, args):
    setup(rank, args.world_size, args.seed)
    main_fn(args, rank)
    cleanup()


def print_loss_log(file_name, train_loss, val_loss, test_loss, args=None):
    with open(file_name, 'w') as f:
        if args:
            for item in args.__dict__:
                f.write(item + ':    ' + str(args.__dict__[item]) + '\n')
        for idx in range(len(train_loss)):
            f.write('epoch {:3d} | train loss {:8.5f}'.format(idx + 1,
                                                              train_loss[idx]) + '\n')
        for idx in range(len(val_loss)):
            f.write('epoch {:3d} | val loss {:8.5f}'.format(idx + 1,
                                                            val_loss[idx]) + '\n')
        f.write('test loss {:8.5f}'.format(test_loss) + '\n')


def wrap_up(train_loss_log, val_loss_log, test_loss, args, model, ns_loss_log, model_filename):
    print('=' * 89)
    print('| End of training | test loss {:8.5f} | test ppl {:8.5f}'.format(test_loss, math.exp(test_loss)))
    print('=' * 89)
    print_loss_log(ns_loss_log, train_loss_log, val_loss_log, test_loss)
    with open(args.save, 'wb') as f:
        torch.save(model.bert_model, f)
    with open(model_filename, 'wb') as f:
        torch.save(model, f)
