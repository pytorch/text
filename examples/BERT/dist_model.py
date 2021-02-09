import threading

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.rpc import RRef

from model import XLMREmbedding, TransformerEncoderLayer, TransformerEncoder


def get_cuda_if_available(i):
    assert i >= 0
    if torch.cuda.is_available():
        return f"cuda:{min(i, torch.cuda.device_count() - 1)}"
    else:
        return "cpu"


class CrossLingualMLMTaskBase(nn.Module):
    def __init__(self, device, underlying):
        super(CrossLingualMLMTaskBase, self).__init__()
        self.device = device
        self.underlying = underlying.to(device)
        self._lock = threading.Lock()

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.underlying(x)
        return out.cpu()

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


class DistCrossLingualMLMTask(nn.Module):
    """Two shards CrossLingualMLMTask"""

    def __init__(self, nshards, split_size, workers, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, *args, **kwargs):
        super(DistCrossLingualMLMTask, self).__init__()

        self.split_size = split_size

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)

        self.shards = []

        if nshards == 2:
            self.shards.append(rpc.remote(
                workers[0],
                CrossLingualMLMTaskBase,
                args=(get_cuda_if_available(0), nn.Sequential(
                    XLMREmbedding(ntoken, ninp, dropout),
                    TransformerEncoder(encoder_layers, nlayers // 2)
                )) + args,
                kwargs=kwargs
            ))

            self.shards.append(rpc.remote(
                workers[1],
                CrossLingualMLMTaskBase,
                args=(get_cuda_if_available(1), nn.Sequential(
                    TransformerEncoder(encoder_layers, nlayers // 2),
                    nn.Linear(ninp, ninp),
                    nn.GELU(),
                    nn.LayerNorm(ninp, eps=1e-12),
                    nn.Linear(ninp, ntoken)
                )) + args,
                kwargs=kwargs
            ))
        elif nshards > 2:
            assert nlayers % (nshards - 2) == 0

            # XLMREmbedding shard
            self.shards.append(rpc.remote(
                workers[0],
                CrossLingualMLMTaskBase,
                args=(get_cuda_if_available(0), XLMREmbedding(ntoken, ninp, dropout)) + args,
                kwargs=kwargs
            ))

            # TODO: encoders
            for s in range(nshards - 2):
                self.shards.append(rpc.remote(
                    workers[s + 1],
                    CrossLingualMLMTaskBase,
                    args=(get_cuda_if_available(s + 1), nn.Sequential(
                        TransformerEncoder(encoder_layers, nlayers // (nshards - 2))
                    )) + args,
                    kwargs=kwargs
                ))

            # MLM Head shard
            self.shards.append(rpc.remote(
                workers[nshards - 1],
                CrossLingualMLMTaskBase,
                args=(get_cuda_if_available(nshards - 1), nn.Sequential(
                    nn.Linear(ninp, ninp),
                    nn.GELU(),
                    nn.LayerNorm(ninp, eps=1e-12),
                    nn.Linear(ninp, ntoken)
                )) + args,
                kwargs=kwargs
            ))

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures into a list
        out_futures = []
        for x in iter(xs.split(self.split_size, dim=0)):
            x_rref = RRef(x)
            for shard in self.shards[:-1]:
                x_rref = shard.remote().forward(x_rref)
            z_fut = self.shards[-1].rpc_async().forward(x_rref)
            out_futures.append(z_fut)

        # collect and cat all output tensors into one tensor.
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        for shard in self.shards:
            remote_params.extend(shard.remote().parameter_rrefs().to_here())
        return remote_params
