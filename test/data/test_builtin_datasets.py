import os
import torchtext.data as data
from torchtext.datasets import WikiText2, PennTreebank

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
