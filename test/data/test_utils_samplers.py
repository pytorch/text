from itertools import combinations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from torchtext.experimental.utils import BucketBatchSampler

from ..common.torchtext_test_case import TorchtextTestCase


class TestSampler(TorchtextTestCase):
    def test_bucket_by_length_batch_sampler(self):
        dummy = [torch.tensor(range(1, torch.randint(2, 11, (1,))[0])) for num in range(15)]

        def tensor_seq_len_fn(row):
            return row.size(0)

        sampler = BucketBatchSampler(dummy, [5, 10], tensor_seq_len_fn, batch_size=5)

        # Ensure all indexes are available from the sampler
        indexes = [idx for row in sampler for idx in row]
        self.assertEquals(sorted(indexes), list(range(15)))

        # Since our bucket boundaries are 5 and 10, we can check if all of the
        # members have no difference more than 5
        def diff_among_members(arr):
            return abs(arr[0] - arr[1])

        for row in sampler:
            lengths = []
            for idx in row:
                lengths.append(dummy[idx].size(0))
            if len(lengths) > 1:
                max_diff = max(combinations(lengths, 2), key=diff_among_members)
                self.assertLess(abs(max_diff[0] - max_diff[1]), 5)

    def test_full_pipeline_bucket_sampler(self):
        class MyDataset(Dataset):
            def __init__(self):
                self.data = []
                for num in range(20):
                    max_length = torch.randint(2, 11, (1,))[0]
                    self.data.append(torch.tensor(range(1, max_length)))

            def __getitem__(self, idx):
                return self.data[idx]

            def __len__(self):
                return len(self.data)

        def collate_fn(batch):
            return pad_sequence(batch, batch_first=True)

        def tensor_seq_len_fn(row):
            return row.size(0)

        dataset = MyDataset()
        batch_sampler = BucketBatchSampler(dataset, [3, 5, 10], tensor_seq_len_fn, 5)
        iterator = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

        for x in iterator:
            mask = x.ne(0)
            seq_lens = mask.sum(-1)
            max_seq_length = seq_lens.max().item()
            min_seq_length = seq_lens.min().item()
            diff = max_seq_length - min_seq_length

            self.assertTrue(diff <= 2 or diff <= 5)
