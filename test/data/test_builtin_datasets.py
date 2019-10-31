import os
import shutil 
import torchtext.data as data
from torchtext.datasets import AG_NEWS

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
        from torchtext.legacy.datasets import WikiText2
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
        if os.environ.get("TRAVIS") == "true":
            datafile = os.path.join(self.project_root, ".data", "wikitext-2")
            conditional_remove(datafile)

    def test_wikitext2(self):
        from torchtext.datasets import WikiText2
        # smoke test to ensure wikitext2 works properly
        train_dataset, test_dataset, valid_dataset = WikiText2()
        self.assertEqual(len(train_dataset), 1947375)
        self.assertEqual(len(test_dataset), 230357)
        self.assertEqual(len(valid_dataset), 203947)

        vocab = train_dataset.get_vocab()
        tokens_ids = [vocab[token] for token in 'the player characters rest'.split()]
        self.assertEqual(tokens_ids, [2, 285, 502, 699])

        # Delete the dataset after we're done to save disk space on CI
        if os.environ.get("TRAVIS") == "true":
            datafile = os.path.join(self.project_root, ".data", "wikitext-2")
            conditional_remove(datafile)
            datafile = os.path.join(self.project_root, ".data", "wikitext-2-v1.zip")
            conditional_remove(datafile)

    @slow
    def test_penntreebank_legacy(self):
        from torchtext.legacy.datasets import PennTreebank
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
        if os.environ.get("TRAVIS") == "true":
            datafile = os.path.join(self.project_root, ".data", "penn-treebank")
            conditional_remove(datafile)

    def test_penntreebank(self):
        from torchtext.datasets import PennTreebank
        # smoke test to ensure wikitext2 works properly
        train_dataset, test_dataset, valid_dataset = PennTreebank()
        self.assertEqual(len(train_dataset), 924412)
        self.assertEqual(len(test_dataset), 82114)
        self.assertEqual(len(valid_dataset), 73339)

        vocab = train_dataset.get_vocab()
        tokens_ids = [vocab[token] for token in 'the player characters rest'.split()]
        self.assertEqual(tokens_ids, [2, 2550, 3344, 1125])

        # Delete the dataset after we're done to save disk space on CI
        if os.environ.get("TRAVIS") == "true":
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

        # Delete the dataset after we're done to save disk space on CI
        if os.environ.get("TRAVIS") == "true":
            datafile = os.path.join(self.project_root, ".data", "ag_news_csv")
            conditional_remove(datafile)
            datafile = os.path.join(self.project_root, ".data", "ag_news_csv.tar.gz")
            conditional_remove(datafile)
