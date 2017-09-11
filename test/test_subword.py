import unittest

from torchtext import data
from torchtext.datasets import TREC, SNLI


class TestSubword(unittest.TestCase):
    def test_subword_trec(self):
        TEXT = data.SubwordField()
        LABEL = data.Field(sequential=False)
        RAW = data.Field(sequential=False, use_vocab=False)
        raw, = TREC.splits(RAW, LABEL, train=TREC.test_filename, test=None)
        cooked, = TREC.splits(TEXT, LABEL, train=TREC.test_filename, test=None)
        LABEL.build_vocab(cooked)
        TEXT.build_vocab(cooked, max_size=100)
        TEXT.segment(cooked)
        print(cooked[0].text)
        batch = next(iter(data.Iterator(cooked, 1, shuffle=False)))
        self.assertEqual(TEXT.reverse(batch.text.data)[0], raw[0].text)

    def test_subword_snli(self):
        TEXT = data.SubwordField()
        LABEL = data.Field(sequential=False)
        RAW = data.Field(sequential=False, use_vocab=False)
        raw, = SNLI.splits(RAW, LABEL, train=None, test=None)
        cooked, = SNLI.splits(TEXT, LABEL, train=None, test=None)
        LABEL.build_vocab(cooked)
        TEXT.build_vocab(cooked, max_size=100)
        TEXT.segment(cooked)
        print(cooked[0].premise)
        print(cooked[0].hypothesis)
        batch = next(iter(data.Iterator(cooked, 1, shuffle=False)))
        self.assertEqual(TEXT.reverse(batch.premise.data)[0], raw[0].premise)
        self.assertEqual(TEXT.reverse(batch.hypothesis.data)[0], raw[0].hypothesis)


if __name__ == '__main__':
    unittest.main()
