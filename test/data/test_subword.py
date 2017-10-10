import unittest

from torchtext import data
from torchtext.datasets import TREC


class TestSubword(unittest.TestCase):
    def test_subword_trec(self):
        TEXT = data.SubwordField()
        LABEL = data.Field(sequential=False)
        RAW = data.Field(sequential=False, use_vocab=False)
        raw, = TREC.splits(RAW, LABEL, train=None)
        cooked, = TREC.splits(TEXT, LABEL, train=None)
        LABEL.build_vocab(cooked)
        TEXT.build_vocab(cooked, max_size=100)
        TEXT.segment(cooked)
        print(cooked[0].text)
        batch = next(iter(data.Iterator(cooked, 1, shuffle=False, device=-1)))
        self.assertEqual(TEXT.reverse(batch.text.data)[0], raw[0].text)


if __name__ == '__main__':
    unittest.main()
