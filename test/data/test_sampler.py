import torch
from torchtext.data.bucket_sampler import BucketSampler
from ..common.torchtext_test_case import TorchtextTestCase


class BucketSamplerTestCase(TorchtextTestCase):
    def test_sampler_overfull_bucket_with_drop(self):
        lengths = [10, 10, 10]
        batch_size = 2

        # Buckets are length [0, 10), [10, 20)
        sampler = iter(BucketSampler(lengths=lengths, buckets=(0, 20, 10), batch_size=batch_size, shuffle=False, drop_last=True))

        sample1 = [lengths[i] for i in next(sampler)]
        self.assertEqual(len(sample1), batch_size)
        self.assertEqual(sample1, torch.tensor([lengths[0], lengths[1]]))

        # Because the last element overflows the bucket and drop_last=True, we drop it
        with self.assertRaises(StopIteration):
            next(sampler)

    def test_sampler_overfull_bucket_without_drop(self):
        lengths = [10, 10, 10]
        batch_size = 2

        # Buckets are length [0, 10), [10, 20)
        sampler = iter(BucketSampler(lengths=lengths, buckets=(0, 20, 10), batch_size=batch_size, shuffle=False, drop_last=False))

        sample1 = [lengths[i] for i in next(sampler)]
        self.assertEqual(len(sample1), batch_size)
        self.assertEqual(sample1, torch.tensor([lengths[0], lengths[1]]))

        sample2 = [lengths[i] for i in next(sampler)]

        self.assertLessEqual(len(sample2), batch_size)
        self.assertEqual(sample2, torch.tensor([lengths[2]]))

    def test_sampler_full_bucket(self):
        lengths = [10, 10, 5]
        batch_size = 2

        # Buckets are length [0, 5), [5, 10), [10, 15), [15, 20)
        sampler = iter(BucketSampler(lengths=lengths, buckets=(0, 20, 5), batch_size=batch_size, shuffle=False, drop_last=False))

        # Take batch_size elements from largest bucket
        sample1 = [lengths[i] for i in next(sampler)]
        self.assertEqual(len(sample1), batch_size)
        self.assertEqual(sample1, torch.tensor([lengths[0], lengths[1]]))

        sample2 = [lengths[i] for i in next(sampler)]
        # because drop_last=False, the sampled batch is smaller than batch_size
        self.assertLessEqual(len(sample2), batch_size)
        self.assertEqual(sample2, torch.tensor([lengths[2]]))
