from heapq import heappop, heappush
from typing import Any, Callable, List, Tuple, Union

import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler


class BucketBatchSampler(Sampler):
    """Defines a batch sampler that batches examples of similar lengths together and
    minimizes amount of padding needed. This BatchSampler works by initially taking a large
    steps (multiplied by 100) and then sort the data according to `seq_len_fn`.
    Arguments:
        data_source: data source to sample from.
        seq_len_fn: function to return the current length of the sequence.
        batch_size: size of mini-batch.
            Default: 32
        shuffle: data_source will be wrapped with RandomSampler if set to ``True``,
            otherwise, SequentialSampler. Default: True
    Example:
        >>> dummy = [
            torch.tensor(range(1, torch.randint(2, 11, (1,))[0])) for num in range(10)
        ]
        >>> def tensor_seq_len_fn(row):
        ...     return row.size(0)
        >>> list(BucketBatchSampler(dummy, tensor_seq_len_fn, batch_size=5, shuffle=False))
        [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9]]
        >>> list(BucketBatchSampler(dummy, tensor_seq_len_fn, batch_size=5))
        [[9, 2, 4, 3, 1], [8, 7, 5, 6], [0]]
    """

    def __init__(
        self,
        data_source: Dataset,
        seq_len_fn: Callable[[Union[List[Any], torch.Tensor]], int],
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        if isinstance(data_source, IterableDataset):
            raise TypeError("Currently does not support IterableDataset!")

        self.data_source = data_source
        self.seq_len_fn = seq_len_fn
        self.batch_size = batch_size
        if shuffle:
            self.sampler = RandomSampler(data_source)
        else:
            self.sampler = SequentialSampler(data_source)

    def __iter__(self):
        sample_count = 100
        minibatch = []
        for idx in self.sampler:
            if len(minibatch) % (self.batch_size * sample_count) == 0:
                for batch in self._batch(minibatch):
                    yield batch
                minibatch = []
            heappush(minibatch, (self.seq_len_fn(self.data_source[idx]), idx))

        # Finish up leftovers
        if minibatch:
            for batch in self._batch(minibatch):
                yield batch

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def _batch(self, minibatch: List[Tuple[torch.Tensor, int]]):
        total_iter = (len(minibatch) + self.batch_size - 1) // self.batch_size
        for _ in range(total_iter):
            max_steps = min(self.batch_size, len(minibatch))
            # Return ordered data
            batch_iter = [heappop(minibatch) for _ in range(max_steps)]
            yield list(map(lambda x: x[1], batch_iter))
