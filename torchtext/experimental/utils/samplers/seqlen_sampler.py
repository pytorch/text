from typing import List

import numpy as np
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler


class BucketByLengthBatchSampler(Sampler):
    """Defines a batch sampler that batches examples of similar lengths together and
    minimizes amount of padding needed.
    Arguments:
        data_source: data source to sample from.
        bucket_boundaries: upper length boundaries to merge sentences with length
            less than or equal to the boundaries.
        batch_size: size of mini-batch.
            Default: 32
        seq_dim: dimension id where the sequence sizes are located.
            Default: 0
        shuffle: data_source will be wrapped with RandomSampler if set to ``True``,
            otherwise, SequentialSampler. Default: True
    Example:
        >>> dummy = [
            torch.tensor(range(1, torch.randint(2, 11, (1,))[0])) for num in range(10)
        ]
        >>> list(BucketByLengthBatchSampler(dummy, [5, 10], batch_size=5, shuffle=False))
        [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9]]
        >>> list(BucketByLengthBatchSampler(dummy, [5, 10], batch_size=5))
        [[9, 2, 4, 3, 1], [8, 7, 5, 6], [0]]
    """

    def __init__(
        self,
        data_source: Dataset,
        bucket_boundaries: List[int],
        batch_size: int = 32,
        seq_dim: int = 0,
        shuffle: bool = True,
    ):
        if isinstance(data_source, IterableDataset):
            raise TypeError("Currently does not support IterableDataset!")

        self.data_source = data_source
        self.seq_dim = seq_dim
        self.bucket_boundaries = bucket_boundaries + [np.inf]
        self.batch_size = batch_size
        if shuffle:
            self.sampler = RandomSampler(data_source)
        else:
            self.sampler = SequentialSampler(data_source)

        self.buckets = []
        for _ in range(len(bucket_boundaries) + 1):
            self.buckets.append([])

    def __iter__(self):
        for idx in self.sampler:
            row = self.data_source[idx]
            for bidx, boundary in enumerate(self.bucket_boundaries):
                if row.size(self.seq_dim) <= boundary:
                    self.buckets[bidx].append(idx)
                    break
            # Flush the buckets
            for bidx, bucket in enumerate(self.buckets):
                if len(bucket) == self.batch_size:
                    yield bucket
                    self.buckets[bidx] = []
        # Flush leftovers
        for bidx, bucket in enumerate(self.buckets):
            if len(bucket) > 0:
                yield bucket

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size
