#!/usr/bin/env python3
# Note that all the tests in this module require dataset (either network access or cached)
import unittest

from torchtext import data
from torchtext.datasets import TREC


class TestSubword(unittest.TestCase):
    def test_subword_trec(self):
        TEXT = data.SubwordField()
        LABEL = data.Field(sequential=False)
        RAW = data.Field(sequential=False, use_vocab=False)
        raw, _ = TREC.splits(RAW, LABEL)
        cooked, _ = TREC.splits(TEXT, LABEL)
        LABEL.build_vocab(cooked)
        TEXT.build_vocab(cooked, max_size=100)
        TEXT.segment(cooked)
        print(cooked[0].text)
        batch = next(iter(data.Iterator(cooked, 1, shuffle=False)))
        self.assertEqual(TEXT.reverse(batch.text.data)[0], raw[0].text)


if __name__ == '__main__':
    unittest.main()
