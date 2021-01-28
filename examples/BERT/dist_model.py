import threading

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.rpc import RRef
from torch.nn import Linear, LayerNorm

from model import XLMREmbedding, TransformerEncoderLayer, TransformerEncoder


def get_cuda_if_available(i):
    assert i >= 0
    if torch.cuda.is_available():
        return f"cuda:{min(i, torch.cuda.device_count() - 1)}"
    else:
        return "cpu"


class CrossLingualMLMTaskBase(nn.Module):
    def __init__(self, device):
        super(CrossLingualMLMTaskBase, self).__init__()
        self.device = device
        self._lock = threading.Lock()

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self._forward(x)
        return out.cpu()

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


class CrossLingualMLMTaskShard1(CrossLingualMLMTaskBase):
    def __init__(self, device, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(CrossLingualMLMTaskShard1, self).__init__(device)
        self.xlmr_embed = XLMREmbedding(ntoken, ninp, dropout).to(device)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers // 2).to(device)

    def _forward(self, src):
        output = self.xlmr_embed(src)
        output = self.transformer_encoder(output)
        return output


class CrossLingualMLMTaskShard2(CrossLingualMLMTaskBase):
    def __init__(self, device, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(CrossLingualMLMTaskShard2, self).__init__(device)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers // 2).to(device)
        self.mlm_span = Linear(ninp, ninp).to(device)
        self.activation = F.gelu
        self.norm_layer = LayerNorm(ninp, eps=1e-12).to(device)
        self.mlm_head = Linear(ninp, ntoken).to(device)

    def _forward(self, src):
        output = self.transformer_encoder(src)
        output = self.mlm_span(output)
        output = self.activation(output)
        output = self.norm_layer(output)
        output = self.mlm_head(output)
        return output


class DistCrossLingualMLMTask(nn.Module):
    """Two shards CrossLingualMLMTask"""

    def __init__(self, split_size, workers, *args, **kwargs):
        super(DistCrossLingualMLMTask, self).__init__()

        self.split_size = split_size

        # Put the first part of the ResNet50 on workers[0]
        self.p1_rref = rpc.remote(
            workers[0],
            CrossLingualMLMTaskShard1,
            args=(get_cuda_if_available(0),) + args,
            kwargs=kwargs
        )

        # Put the second part of the ResNet50 on workers[1]
        self.p2_rref = rpc.remote(
            workers[1],
            CrossLingualMLMTaskShard2,
            args=(get_cuda_if_available(1),) + args,
            kwargs=kwargs
        )

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures into a list
        out_futures = []
        for x in iter(xs.split(self.split_size, dim=0)):
            x_rref = RRef(x)
            y_rref = self.p1_rref.remote().forward(x_rref)
            z_fut = self.p2_rref.rpc_async().forward(y_rref)
            out_futures.append(z_fut)

        # collect and cat all output tensors into one tensor.
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        return remote_params
