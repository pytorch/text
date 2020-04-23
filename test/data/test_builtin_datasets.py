import os
import shutil
import torchtext.data as data
from torchtext.datasets import AG_NEWS
import torch
from torch.testing import assert_allclose
from ..common.test_markers import slow
from ..common.torchtext_test_case import TorchtextTestCase


def conditional_remove(f):
    if os.path.isfile(f):
        os.remove(f)
    elif os.path.isdir(f):
        shutil.rmtree(f)


class TestDataset(TorchtextTestCase):
    @slow
    def test_wikitext2_legacy(self):
        from torchtext.datasets import WikiText2
        # smoke test to ensure wikitext2 works properly
        ds = WikiText2
        TEXT = data.Field(lower=True, batch_first=True)
        train, valid, test = ds.splits(TEXT)
        TEXT.build_vocab(train)
        train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
            (train, valid, test), batch_size=3, bptt_len=30)

        train_iter, valid_iter, test_iter = ds.iters(batch_size=4,
                                                     bptt_len=30)

        # Delete the dataset after we're done to save disk space on CI
        datafile = os.path.join(self.project_root, ".data", "wikitext-2")
        conditional_remove(datafile)

    def test_wikitext2(self):
        from torchtext.experimental.datasets import WikiText2
        # smoke test to ensure wikitext2 works properly
        train_dataset, test_dataset, valid_dataset = WikiText2()
        self.assertEqual(len(train_dataset), 2049990)
        self.assertEqual(len(test_dataset), 241859)
        self.assertEqual(len(valid_dataset), 214417)

        vocab = train_dataset.get_vocab()
        tokens_ids = [vocab[token] for token in 'the player characters rest'.split()]
        self.assertEqual(tokens_ids, [2, 286, 503, 700])

        # Delete the dataset after we're done to save disk space on CI
        datafile = os.path.join(self.project_root, ".data", "wikitext-2")
        conditional_remove(datafile)
        datafile = os.path.join(self.project_root, ".data", "wikitext-2-v1.zip")
        conditional_remove(datafile)

    @slow
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

        # Delete the dataset after we're done to save disk space on CI
        datafile = os.path.join(self.project_root, ".data", "penn-treebank")
        conditional_remove(datafile)

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

        # Delete the dataset after we're done to save disk space on CI
        datafile = os.path.join(self.project_root, ".data", 'ptb.train.txt')
        conditional_remove(datafile)
        datafile = os.path.join(self.project_root, ".data", 'ptb.test.txt')
        conditional_remove(datafile)
        datafile = os.path.join(self.project_root, ".data", 'ptb.valid.txt')
        conditional_remove(datafile)

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
                        torch.tensor(tensor([2351, 758, 96, 38581, 2351, 220, 5, 396, 3, 14786])).long())

        # Delete the dataset after we're done to save disk space on CI
        datafile = os.path.join(self.project_root, ".data", "ag_news_csv")
        conditional_remove(datafile)
        datafile = os.path.join(self.project_root, ".data", "ag_news_csv.tar.gz")
        conditional_remove(datafile)

    @slow
    def test_imdb(self):
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

        # Delete the dataset after we're done to save disk space on CI
        datafile = os.path.join(self.project_root, ".data", "imdb")
        conditional_remove(datafile)
        datafile = os.path.join(self.project_root, ".data", "aclImdb")
        conditional_remove(datafile)
        datafile = os.path.join(self.project_root, ".data", "aclImdb_v1.tar.gz")
        conditional_remove(datafile)
