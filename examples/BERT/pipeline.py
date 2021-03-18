import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import threading
import concurrent.futures

class ToDevice(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, x):
        return x.to(self.device)


class SingleProcessPipeline(nn.Sequential):
    def __init__(self, shards, devices):
        super().__init__()
        assert len(shards) == len(devices)
        self.devices = devices
        self.seq = nn.Sequential()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            concurrent.futures.wait([executor.submit(lambda s, d: s.to(d), shards[i], devices[i]) for i in range(len(shards))])

        for i, shard in enumerate(shards):
            self.seq.add_module(f'Shard({devices[i]})', shard)
            if i != len(shards)-1:
                self.seq.add_module(f'ToDevice({devices[i+1]})', ToDevice(devices[i+1]))


class RemoteBaseCPURPC(nn.Module):
    def __init__(self, underlying, device):
        super().__init__()
        self.underlying = underlying.to(device)
        self.device = device
        self._lock = threading.Lock()

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.underlying(x)
        return out.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


class RemoteBaseCUDARPC(nn.Module):
    def __init__(self, underlying, device):
        super().__init__()
        self.underlying = underlying.to(device)
        self.device = device
        self._lock = threading.Lock()

    def forward(self, x_rref):
        with self._lock:
            return self.underlying(x_rref.to_here())

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


class RPCPipeline(nn.Module):
    def __init__(self, shards, devices, workers, remote_base_class=RemoteBaseCPURPC, split_size=1):
        super().__init__()
        self.split_size = split_size
        self.shards = [rpc.remote(worker, remote_base_class, args=(shard, device)) for worker, shard, device in zip(workers, shards, devices)]

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

# =======================================================================================================================================

class MyPipelineLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.underlying_moved_to_device = False

    def move_underlying_to_device():
        pass


class MyPipelineLocalMultiGPULayer(MyPipelineLayer):
    def __init__(self, layer, in_features, out_features, devices=None, result_device=None, *args, **kwargs):
        super().__init__()
        self.devices = devices
        self.result_device = result_device

        if self.devices is None:
            self.devices = ['cpu']
        if self.result_device is None:
            self.result_device = self.devices[-1]

        n_devices = len(self.devices)
        if out_features < n_devices:
            self.shard_out_features = [1] * out_features
            self.devices = self.devices[:out_features]
        elif out_features % n_devices == 0:
            self.shard_out_features = [out_features // n_devices] * n_devices
        else:
            size = (out_features + n_devices - 1) // n_devices
            self.shard_out_features = [size] * (n_devices - 1) + [out_features - size * (n_devices - 1)]
        self.linears = nn.Sequential(*[layer(in_features, shard_out_feature, *args, **kwargs) for shard_out_feature in self.shard_out_features])

    def move_underlying_to_device(self):
        for device, linear in zip(self.devices, self.linears):
            linear.to(device)

    def forward(self, input):
        return torch.cat([linear(input.to(device)).to(self.result_device) for device, linear in zip(self.devices, self.linears)], dim=-1)


class MyPipelineWrapper(MyPipelineLayer):
    def __init__(self, underlying, device):
        super().__init__()
        self.underlying = underlying
        self.device = device

    def move_underlying_to_device(self):
        self.underlying.to(self.device)
        self.underlying_moved_to_device = True

    def _move_input_to_device(self, *args, **kwargs):
        new_args = tuple(arg.to(self.device) if isinstance(arg, torch.Tensor) else arg for arg in args)
        new_kwargs = {key: value.to(device) if isinstance(value, torch.Tensor) else  value for key, value in kwargs}
        return new_args, new_kwargs

    def forward(self, *args, **kwargs):
        args, kwargs = self._move_input_to_device(*args, **kwargs)
        return self.underlying(*args, **kwargs)


class MyPipeline(nn.Sequential):
    def __init__(self, *layers):
        super().__init__(*layers)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            concurrent.futures.wait([executor.submit(lambda l: l.move_underlying_to_device(), layer) for layer in self])

# =======================================================================================================================================

class MyRPCPipelineWrapper(nn.Module):
    def __init__(self, underlying, remote_device):
        super().__init__()
        self.underlying = underlying
        self.worker, self.device = remote_device.split(":")
        self.device = int(self.device)
        self.shard = None

    def move_underlying_to_device(self):
        self.shard = rpc.remote(self.worker, RemoteBaseCPURPC, args=(self.underlying, self.device))
        self.underlying_moved_to_device = True

    def forward(self, *args, **kwargs):
        return self.shard.remote().forward(*args, **kwargs)

    def parameter_rrefs(self):
        return self.shard.remote().parameter_rrefs()


class MyRPCPipelineDistMultiGPULayer(nn.Module):
    def __init__(self, layer, in_features, out_features, remote_devices=None, result_remote_device=None, *args, **kwargs):
        super().__init__()
        self.workers = [remote_device.split(":")[0] for remote_device in remote_devices]
        self.devices = [int(remote_device.split(":")[1]) for remote_device in remote_devices]
        if result_remote_device is None:
            result_remote_device = remote_devices[-1]
        self.result_worker, self.result_device = result_remote_device.split(":")
        self.result_device = int(self.result_device)

        n_devices = len(remote_devices)
        if out_features < n_devices:
            self.shard_out_features = [1] * out_features
            self.workers = self.workers[:out_features]
            self.devices = self.devices[:out_features]
        elif out_features % n_devices == 0:
            self.shard_out_features = [out_features // n_devices] * n_devices
        else:
            size = (out_features + n_devices - 1) // n_devices
            self.shard_out_features = [size] * (n_devices - 1) + [out_features - size * (n_devices - 1)]
        self.in_features = in_features
        # TODO: move to move_underlying_to_device
        self.linears = [rpc.remote(worker, RemoteBaseCPURPC, args=(layer(self.in_features, shard_out_feature, *args, **kwargs), device)) for worker, shard_out_feature, device in zip(self.workers, self.shard_out_features, self.devices)]

    def move_underlying_to_device(self):
        pass

    def forward(self, input):
        return RRef(torch.cat([linear.remote().forward(input).to_here() for linear in self.linears], dim=-1))

    def parameter_rrefs(self):
        remote_params = []
        for layer in self.linears:
            remote_params.extend(layer.remote().parameter_rrefs().to_here())
        return RRef(remote_params)


class MyRPCPipeline(nn.Sequential):
    def __init__(self, *layers):
        super().__init__(*layers)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            concurrent.futures.wait([executor.submit(lambda l: l.move_underlying_to_device(), layer) for layer in self])

    def forward(self, x):
        return super().forward(RRef(x)).to_here()

    def parameter_rrefs(self):
        remote_params = []
        for layer in self:
            remote_params.extend(layer.parameter_rrefs().to_here())
        return remote_params
