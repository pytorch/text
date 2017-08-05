import unittest

from torchtext import data
from torchtext.datasets import TREC


class TestSubword(unittest.TestCase):
    def test_subword(self):
        TEXT = data.SubwordField()
        LABEL = data.Field(sequential=False)
        RAW = data.Field(sequential=False, use_vocab=False)
        raw, = TREC.splits(RAW, LABEL, train=TREC.test_filename, test=None)
        cooked, = TREC.splits(TEXT, LABEL, train=TREC.test_filename, test=None)
        LABEL.build_vocab(cooked)
        TEXT.build_vocab(cooked, max_size=100)
        TEXT.segment(cooked)
        batch = next(iter(data.Iterator(cooked, 1, shuffle=False)))
        self.assertEqual(TEXT.reverse(batch.text.data)[0], raw[0].text)


if __name__ == '__main__':
    unittest.main()
