import torch
from torch.testing import assert_allclose

from ..common.torchtext_test_case import TorchtextTestCase


class TestDataset(TorchtextTestCase):
    def test_imdb(self):
        # This test requires network access if no cache is found in `.data` dir
        from torchtext.experimental.datasets import IMDB
        from torchtext.vocab import Vocab
        # smoke test to ensure imdb works properly
        train_dataset, test_dataset = IMDB()
        self.assertEqual(len(train_dataset), 25000)
        self.assertEqual(len(test_dataset), 25000)
        assert_allclose(train_dataset[0][1][:10],
                        torch.tensor([13, 1568, 13, 246, 35468, 43, 64, 398, 1135, 92]).long())
        assert_allclose(train_dataset[-1][1][:10],
                        torch.tensor([2, 71, 4555, 194, 3328, 15144, 42, 227, 148, 8]).long())
        assert_allclose(test_dataset[0][1][:10],
                        torch.tensor([13, 125, 1051, 5, 246, 1652, 8, 277, 66, 20]).long())
        assert_allclose(test_dataset[-1][1][:10],
                        torch.tensor([13, 1035, 14, 21, 28, 2, 1051, 1275, 1008, 3]).long())

        # Test API with a vocab input object
        old_vocab = train_dataset.get_vocab()
        new_vocab = Vocab(counter=old_vocab.freqs, max_size=2500)
        new_train_data, new_test_data = IMDB(vocab=new_vocab)
