import os
import torchtext.data as data
from torchtext.datasets import WikiText2, PennTreebank, AG_NEWS, ATIS

from ..common.test_markers import slow
from ..common.torchtext_test_case import TorchtextTestCase


def conditional_remove(f):
    if os.path.isfile(f):
        os.remove(f)


class TestDataset(TorchtextTestCase):
    @slow
    def test_wikitext2(self):
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

    @slow
    def test_penntreebank(self):
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
            datafile = os.path.join(self.project_root, ".data", "AG_NEWS")
            conditional_remove(datafile)

    @slow
    def test_atis(self):
        # smoke test to ensure ATIS works properly
        TEXT = data.Field(lower=True, batch_first=True)
        SLOT = data.Field(batch_first=True, unk_token=None)
        INTENT = data.Field(batch_first=True, unk_token=None)

        # make splits for data
        train, val, test = ATIS.splits(
            fields=(('text', TEXT), ('slot', SLOT), ('intent', INTENT)))

        TEXT.build_vocab(train)
        SLOT.build_vocab(train, val, test)
        INTENT.build_vocab(train, val, test)

        train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (train, val, test), batch_sizes=(32, 256, 256))

        # Delete the dataset after we're done to save disk space on CI
        if os.environ.get("TRAVIS") == "true":
            datafile = os.path.join(self.project_root, ".data", "atis")
            conditional_remove(datafile)
