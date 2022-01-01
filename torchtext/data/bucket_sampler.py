import random
from collections import defaultdict

import torch
from typing import Tuple, Iterable

# (Min bucket size, max bucket size, step-size between buckets)
BucketRange = Tuple[int, int, int]


class BucketSampler(torch.utils.data.Sampler):
    # Inspired by https://github.com/pytorch/pytorch/issues/46176#issue-719202723
    def __init__(
            self, lengths: Iterable[int], buckets: BucketRange = (50, 500, 50), shuffle: bool = True,
            batch_size: int = 32, drop_last: bool = False
    ):

        super().__init__(lengths)

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last

        assert isinstance(buckets, tuple)
        bmin, bmax, bstep = buckets
        assert (bmax - bmin) % bstep == 0

        buckets = defaultdict(list)
        for i, length in enumerate(lengths):
            if length > bmin:
                bucket_size = min((length // bstep) * bstep, bmax)
                buckets[bucket_size].append(i)

        self.buckets = dict()
        for bucket_size, bucket in buckets.items():
            if len(bucket) > 0:
                self.buckets[bucket_size] = torch.tensor(bucket, dtype=torch.int32, device='cpu')

        # call __iter__() to store self.length
        self.__iter__()

    def __iter__(self):
        if self.shuffle:
            for bucket_size in self.buckets.keys():
                self.buckets[bucket_size] = self.buckets[bucket_size][
                    torch.randperm(self.buckets[bucket_size].nelement())]

        batches = []
        for bucket in self.buckets.values():
            curr_bucket = torch.split(bucket, self.batch_size)
            if len(curr_bucket) > 1 and self.drop_last:
                if len(curr_bucket[-1]) < len(curr_bucket[-2]):
                    curr_bucket = curr_bucket[:-1]
            batches += curr_bucket

        self.length = len(batches)

        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return self.length
