#!/user/bin/env python3
# Note that all the tests in this module require dataset (either network access or cached)
import os
import glob
import shutil
import torchtext.data as data
from torchtext.datasets import AG_NEWS
import torch
from ..common.torchtext_test_case import TorchtextTestCase


def conditional_remove(f):
    for path in glob.glob(f):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


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
        # smoke test to ensure penn treebank works properly
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
        self.assertEqual(ag_news_train[-1][1][:10],
                         torch.tensor([3525, 319, 4053, 34, 5407, 3607, 70, 6798, 10599, 4053]).long())
        self.assertEqual(ag_news_test[-1][1][:10],
                         torch.tensor([2351, 758, 96, 38581, 2351, 220, 5, 396, 3, 14786]).long())

    def test_imdb(self):
        from torchtext.experimental.datasets import IMDB
        from torchtext.vocab import Vocab
        # smoke test to ensure imdb works properly
        train_dataset, test_dataset = IMDB()
        self.assertEqual(len(train_dataset), 25000)
        self.assertEqual(len(test_dataset), 25000)
        self.assertEqual(train_dataset[0][1][:10],
                         torch.tensor([13, 1568, 13, 246, 35468, 43, 64, 398, 1135, 92]).long())
        self.assertEqual(train_dataset[-1][1][:10],
                         torch.tensor([2, 71, 4555, 194, 3328, 15144, 42, 227, 148, 8]).long())
        self.assertEqual(test_dataset[0][1][:10],
                         torch.tensor([13, 125, 1051, 5, 246, 1652, 8, 277, 66, 20]).long())
        self.assertEqual(test_dataset[-1][1][:10],
                         torch.tensor([13, 1035, 14, 21, 28, 2, 1051, 1275, 1008, 3]).long())

        # Test API with a vocab input object
        old_vocab = train_dataset.get_vocab()
        new_vocab = Vocab(counter=old_vocab.freqs, max_size=2500)
        new_train_data, new_test_data = IMDB(vocab=new_vocab)

    def test_multi30k(self):
        from torchtext.experimental.datasets.translation import Multi30k
        # smoke test to ensure multi30k works properly
        train_dataset, valid_dataset, test_dataset = Multi30k()
        self.assertEqual(len(train_dataset), 29000)
        self.assertEqual(len(valid_dataset), 1000)
        self.assertEqual(len(test_dataset), 1014)

        de_vocab, en_vocab = train_dataset.get_vocab()
        de_tokens_ids = [
            de_vocab[token] for token in
            'Zwei Männer verpacken Donuts in Kunststofffolie'.split()
        ]
        self.assertEqual(de_tokens_ids, [19, 29, 18703, 4448, 5, 6240])

        en_tokens_ids = [
            en_vocab[token] for token in
            'Two young White males are outside near many bushes'.split()
        ]
        self.assertEqual(en_tokens_ids,
                         [17, 23, 1167, 806, 15, 55, 82, 334, 1337])

        datafile = os.path.join(self.project_root, ".data", "train*")
        conditional_remove(datafile)
        datafile = os.path.join(self.project_root, ".data", "val*")
        conditional_remove(datafile)
        datafile = os.path.join(self.project_root, ".data", "test*")
        conditional_remove(datafile)
        datafile = os.path.join(self.project_root, ".data",
                                "multi30k_task*.tar.gz")
        conditional_remove(datafile)

    def test_squad1(self):
        from torchtext.experimental.datasets import SQuAD1
        from torchtext.vocab import Vocab
        # smoke test to ensure imdb works properly
        train_dataset, dev_dataset = SQuAD1()
        self.assertEqual(len(train_dataset), 87599)
        self.assertEqual(len(dev_dataset), 10570)
        self.assertEqual(train_dataset[100]['question'],
                         torch.tensor([7, 24, 86, 52, 2, 373, 887, 18, 12797, 11090, 1356, 2, 1788, 3273, 16]).long())
        self.assertEqual(train_dataset[100]['ans_pos'][0],
                         torch.tensor([72, 72]).long())
        self.assertEqual(dev_dataset[100]['question'],
                         torch.tensor([42, 27, 669, 7438, 17, 2, 1950, 3273, 17252, 389, 16]).long())
        self.assertEqual(dev_dataset[100]['ans_pos'][0],
                         torch.tensor([45, 48]).long())

        # Test API with a vocab input object
        old_vocab = train_dataset.get_vocab()
        new_vocab = Vocab(counter=old_vocab.freqs, max_size=2500)
        new_train_data, new_test_data = SQuAD1(vocab=new_vocab)

    def test_squad2(self):
        from torchtext.experimental.datasets import SQuAD2
        from torchtext.vocab import Vocab
        # smoke test to ensure imdb works properly
        train_dataset, dev_dataset = SQuAD2()
        self.assertEqual(len(train_dataset), 130319)
        self.assertEqual(len(dev_dataset), 11873)
        self.assertEqual(train_dataset[200]['question'],
                         torch.tensor([84, 50, 1421, 12, 5439, 4569, 17, 30, 2, 15202, 4754, 1421, 16]).long())
        self.assertEqual(train_dataset[200]['ans_pos'][0],
                         torch.tensor([9, 9]).long())
        self.assertEqual(dev_dataset[200]['question'],
                         torch.tensor([41, 29, 2, 66, 17016, 30, 0, 1955, 16]).long())
        self.assertEqual(dev_dataset[200]['ans_pos'][0],
                         torch.tensor([40, 46]).long())

        # Test API with a vocab input object
        old_vocab = train_dataset.get_vocab()
        new_vocab = Vocab(counter=old_vocab.freqs, max_size=2500)
        new_train_data, new_test_data = SQuAD2(vocab=new_vocab)
