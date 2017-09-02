# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from collections import Counter
import unittest

import numpy as np
from numpy.testing import assert_allclose
from torchtext import vocab


class TestVocab(unittest.TestCase):
    def test_vocab_basic(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        v = vocab.Vocab(c, min_freq=3, specials=['<pad>', '<bos>'])

        self.assertEqual(v.itos, ['<unk>', '<pad>', '<bos>',
                                  'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world'])

    def test_vocab_download_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        v = vocab.Vocab(c, min_freq=3, specials=['<pad>', '<bos>'],
                        vectors='fasttext.simple.300d')

        self.assertEqual(v.itos, ['<unk>', '<pad>', '<bos>',
                                  'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world'])
        vectors = v.vectors.numpy()

        # The first 5 entries in each vector.
        expected_glove_twitter = {
            'hello': [0.39567, 0.21454, -0.035389, -0.24299, -0.095645],
            'world': [0.10444, -0.10858, 0.27212, 0.13299, -0.33165],
        }

        for word in expected_glove_twitter:
            assert_allclose(vectors[v.stoi[word], :5], expected_glove_twitter[word])

        assert_allclose(vectors[v.stoi['<unk>']], np.zeros(300))
