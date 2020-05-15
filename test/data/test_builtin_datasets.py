import os
import shutil
import torchtext.data as data
from torchtext.datasets import AG_NEWS
import torch
from torch.testing import assert_allclose
from ..common.torchtext_test_case import TorchtextTestCase


def conditional_remove(f):
    if os.path.isfile(f):
        os.remove(f)
    elif os.path.isdir(f):
        shutil.rmtree(f)


class TestDataset(TorchtextTestCase):
    def test_wikitext2_legacy(self):
        from torchtext.datasets import WikiText2
        # smoke test to ensure wikitext2 works properly

        # NOTE
        # test_wikitext2 and test_wikitext2_legacy have some cache incompatibility.
        # Keeping one's cache make the other fail. So we need to clean up the cache dir
        cachedir = os.path.join(self.project_root, ".data", "wikitext-2")
        conditional_remove(cachedir)

        ds = WikiText2
        TEXT = data.Field(lower=True, batch_first=True)
        train, valid, test = ds.splits(TEXT)
        TEXT.build_vocab(train)
        train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
            (train, valid, test), batch_size=3, bptt_len=30)

        train_iter, valid_iter, test_iter = ds.iters(batch_size=4,
                                                     bptt_len=30)

        conditional_remove(cachedir)

    def test_wikitext2(self):
        from torchtext.experimental.datasets import WikiText2
        # smoke test to ensure wikitext2 works properly

        # NOTE
        # test_wikitext2 and test_wikitext2_legacy have some cache incompatibility.
        # Keeping one's cache make the other fail. So we need to clean up the cache dir
        cachedir = os.path.join(self.project_root, ".data", "wikitext-2")
        conditional_remove(cachedir)
        cachefile = os.path.join(self.project_root, ".data", "wikitext-2-v1.zip")
        conditional_remove(cachefile)

        train_dataset, test_dataset, valid_dataset = WikiText2()
        self.assertEqual(len(train_dataset), 2049990)
        self.assertEqual(len(test_dataset), 241859)
        self.assertEqual(len(valid_dataset), 214417)

        vocab = train_dataset.get_vocab()
        tokens_ids = [vocab[token] for token in 'the player characters rest'.split()]
        self.assertEqual(tokens_ids, [2, 286, 503, 700])

        conditional_remove(cachedir)
        conditional_remove(cachefile)

    def test_penntreebank_legacy(self):
        from torchtext.datasets import PennTreebank
        # smoke test to ensure penn treebank works properly
        TEXT = data.Field(lower=True, batch_first=True)
        ds = PennTreebank
        train, valid, test = ds.splits(TEXT)
        TEXT.build_vocab(train)
        train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
            (train, valid, test), batch_size=3, bptt_len=30)

        train_iter, valid_iter, test_iter = ds.iters(batch_size=4,
                                                     bptt_len=30)

    def test_penntreebank(self):
        from torchtext.experimental.datasets import PennTreebank
        # smoke test to ensure wikitext2 works properly
        train_dataset, test_dataset, valid_dataset = PennTreebank()
        self.assertEqual(len(train_dataset), 924412)
        self.assertEqual(len(test_dataset), 82114)
        self.assertEqual(len(valid_dataset), 73339)

        vocab = train_dataset.get_vocab()
        tokens_ids = [vocab[token] for token in 'the player characters rest'.split()]
        self.assertEqual(tokens_ids, [2, 2550, 3344, 1125])

    def test_text_classification(self):
        # smoke test to ensure ag_news dataset works properly

        datadir = os.path.join(self.project_root, ".data")
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        ag_news_train, ag_news_test = AG_NEWS(root=datadir, ngrams=3)
        self.assertEqual(len(ag_news_train), 120000)
        self.assertEqual(len(ag_news_test), 7600)
        assert_allclose(ag_news_train[-1][1][:10],
                        torch.tensor([3525, 319, 4053, 34, 5407, 3607, 70, 6798, 10599, 4053]).long())
        assert_allclose(ag_news_test[-1][1][:10],
                        torch.tensor([2351, 758, 96, 38581, 2351, 220, 5, 396, 3, 14786]).long())
