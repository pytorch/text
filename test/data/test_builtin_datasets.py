#!/user/bin/env python3
# Note that all the tests in this module require dataset (either network access or cached)
import os
import glob
import shutil
import torchtext.data as data
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
