# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from collections import Counter
import logging
import unittest

import numpy as np
from numpy.testing import assert_allclose
from torchtext import vocab

from .common.test_markers import slow

logging.basicConfig(format="%(asctime)s - %(levelname)s "
                    "- %(name)s - %(message)s",
                    level=logging.INFO)


class TestVocab(unittest.TestCase):
    def test_vocab_basic(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        v = vocab.Vocab(c, min_freq=3, specials=['<pad>', '<bos>'])

        self.assertEqual(v.itos, ['<unk>', '<pad>', '<bos>',
                                  'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world'])

    def test_vocab_download_fasttext_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching.
        for i in range(2):
            v = vocab.Vocab(c, min_freq=3, specials=['<pad>', '<bos>'],
                            vectors='fasttext.simple.300d')

            self.assertEqual(v.itos, ['<unk>', '<pad>', '<bos>',
                                      'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world'])
            vectors = v.vectors.numpy()

            # The first 5 entries in each vector.
            expected_fasttext_simple_en = {
                'hello': [0.39567, 0.21454, -0.035389, -0.24299, -0.095645],
                'world': [0.10444, -0.10858, 0.27212, 0.13299, -0.33165],
            }

            for word in expected_fasttext_simple_en:
                assert_allclose(vectors[v.stoi[word], :5],
                                expected_fasttext_simple_en[word])

            assert_allclose(vectors[v.stoi['<unk>']], np.zeros(300))

    @slow
    def test_vocab_download_glove_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching.
        for i in range(2):
            v = vocab.Vocab(c, min_freq=3, specials=['<pad>', '<bos>'],
                            vectors='glove.twitter.27B.25d')

            self.assertEqual(v.itos, ['<unk>', '<pad>', '<bos>',
                                      'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world'])
            vectors = v.vectors.numpy()

            # The first 5 entries in each vector.
            expected_twitter = {
                'hello': [-0.77069, 0.12827, 0.33137, 0.0050893, -0.47605],
                'world': [0.10301, 0.095666, -0.14789, -0.22383, -0.14775],
            }

            for word in expected_twitter:
                assert_allclose(vectors[v.stoi[word], :5],
                                expected_twitter[word])

            assert_allclose(vectors[v.stoi['<unk>']], np.zeros(25))

    @slow
    def test_vocab_download_charngram_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching.
        for i in range(2):
            v = vocab.Vocab(c, min_freq=3, specials=['<pad>', '<bos>'],
                            vectors='charngram.100d')

            self.assertEqual(v.itos, ['<unk>', '<pad>', '<bos>',
                                      'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world'])
            vectors = v.vectors.numpy()

            # The first 5 entries in each vector.
            expected_charngram = {
                'hello': [-0.44782442, -0.08937783, -0.34227219,
                          -0.16233221, -0.39343098],
                'world': [-0.29590717, -0.05275926, -0.37334684, 0.27117205, -0.3868292],
            }

            for word in expected_charngram:
                assert_allclose(vectors[v.stoi[word], :5],
                                expected_charngram[word])

            assert_allclose(vectors[v.stoi['<unk>']], np.zeros(100))
