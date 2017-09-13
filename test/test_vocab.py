# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from collections import Counter
import os
import pickle


import numpy as np
from numpy.testing import assert_allclose
import torch
from torchtext import vocab
from torchtext.vocab import Vectors, FastText, GloVe, CharNGram

from .common.test_markers import slow
from .common.torchtext_test_case import TorchtextTestCase


def conditional_remove(f):
    if os.path.isfile(f):
        os.remove(f)


class TestVocab(TorchtextTestCase):

    def test_vocab_basic(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        v = vocab.Vocab(c, min_freq=3, specials=['<pad>', '<bos>'])

        expected_itos = ['<unk>', '<pad>', '<bos>',
                         'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world']
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}
        self.assertEqual(v.itos, expected_itos)
        self.assertEqual(dict(v.stoi), expected_stoi)

    def test_vocab_set_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5,
                     'ｔｅｓｔ': 4, 'freq_too_low': 2})
        v = vocab.Vocab(c, min_freq=3, specials=['<pad>', '<bos>'])
        stoi = {"hello": 0, "world": 1, "ｔｅｓｔ": 2}
        vectors = torch.FloatTensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        dim = 2
        v.set_vectors(stoi, vectors, dim)
        expected_vectors = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                     [0.0, 0.0], [0.1, 0.2], [0.5, 0.6],
                                     [0.3, 0.4]])
        assert_allclose(v.vectors.numpy(), expected_vectors)

    def test_vocab_download_fasttext_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching, then once more
        # to test string aliases.
        for i in range(3):
            if i == 2:
                vectors = "fasttext.simple.300d"
            else:
                vectors = FastText(language='simple')

            v = vocab.Vocab(c, min_freq=3, specials=['<pad>', '<bos>'],
                            vectors=vectors)

            expected_itos = ['<unk>', '<pad>', '<bos>',
                             'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world']
            expected_stoi = {x: index for index, x in enumerate(expected_itos)}
            self.assertEqual(v.itos, expected_itos)
            self.assertEqual(dict(v.stoi), expected_stoi)
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
            assert_allclose(vectors[v.stoi['OOV token']], np.zeros(300))
        # Delete the vectors after we're done to save disk space on CI
        if os.environ.get("TRAVIS") == "true":
            vec_file = os.path.join(self.project_root, ".vector_cache", "wiki.simple.vec")
            conditional_remove(vec_file)

    def test_vocab_extend(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching.
        for i in range(2):
            f = FastText(language='simple')
            v = vocab.Vocab(c, min_freq=3, specials=['<pad>', '<bos>'],
                            vectors=f)
            n_vocab = len(v)
            v.extend(f)  # extend the vocab with the words contained in f.itos
            self.assertGreater(len(v), n_vocab)

            self.assertEqual(v.itos[:6], ['<unk>', '<pad>', '<bos>',
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
        # Delete the vectors after we're done to save disk space on CI
        if os.environ.get("TRAVIS") == "true":
            vec_file = os.path.join(self.project_root, ".vector_cache", "wiki.simple.vec")
            conditional_remove(vec_file)

    def test_vocab_download_custom_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching.
        for i in range(2):
            v = vocab.Vocab(c, min_freq=3, specials=['<pad>', '<bos>'],
                            vectors=Vectors('wiki.simple.vec',
                                            url=FastText.url_base.format('simple')))

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
        # Delete the vectors after we're done to save disk space on CI
        if os.environ.get("TRAVIS") == "true":
            vec_file = os.path.join(self.project_root, ".vector_cache", "wiki.simple.vec")
            conditional_remove(vec_file)

    @slow
    def test_vocab_download_glove_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})

        # Build a vocab and get vectors twice to test caching, then once more
        # to test string aliases.
        for i in range(3):
            if i == 2:
                vectors = "glove.twitter.27B.25d"
            else:
                vectors = GloVe(name='twitter.27B', dim='25')
            v = vocab.Vocab(c, min_freq=3, specials=['<pad>', '<bos>'],
                            vectors=vectors)

            expected_itos = ['<unk>', '<pad>', '<bos>',
                             'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world']
            expected_stoi = {x: index for index, x in enumerate(expected_itos)}
            self.assertEqual(v.itos, expected_itos)
            self.assertEqual(dict(v.stoi), expected_stoi)

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
            assert_allclose(vectors[v.stoi['OOV token']], np.zeros(25))
        # Delete the vectors after we're done to save disk space on CI
        if os.environ.get("TRAVIS") == "true":
            zip_file = os.path.join(self.project_root, ".vector_cache",
                                    "glove.twitter.27B.zip")
            conditional_remove(zip_file)
            for dim in ["25", "50", "100", "200"]:
                conditional_remove(os.path.join(self.project_root, ".vector_cache",
                                   "glove.twitter.27B.{}d.txt".format(dim)))

    @slow
    def test_vocab_download_charngram_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching, then once more
        # to test string aliases.
        for i in range(3):
            if i == 2:
                vectors = "charngram.100d"
            else:
                vectors = CharNGram()
            v = vocab.Vocab(c, min_freq=3, specials=['<pad>', '<bos>'],
                            vectors=vectors)
            expected_itos = ['<unk>', '<pad>', '<bos>',
                             'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world']
            expected_stoi = {x: index for index, x in enumerate(expected_itos)}
            self.assertEqual(v.itos, expected_itos)
            self.assertEqual(dict(v.stoi), expected_stoi)
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
            assert_allclose(vectors[v.stoi['OOV token']], np.zeros(100))
        # Delete the vectors after we're done to save disk space on CI
        if os.environ.get("TRAVIS") == "true":
            conditional_remove(
                os.path.join(self.project_root, ".vector_cache", "charNgram.txt"))
            conditional_remove(
                os.path.join(self.project_root, ".vector_cache",
                             "jmt_pre-trained_embeddings.tar.gz"))

    def test_errors(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        with self.assertRaises(ValueError):
            # Test proper error raised when using unknown string alias
            vocab.Vocab(c, min_freq=3, specials=['<pad>', '<bos>'],
                        vectors=["fasttext.english.300d"])
            vocab.Vocab(c, min_freq=3, specials=['<pad>', '<bos>'],
                        vectors="fasttext.english.300d")
        with self.assertRaises(ValueError):
            # Test proper error is raised when vectors argument is
            # non-string or non-Vectors
            vocab.Vocab(c, min_freq=3, specials=['<pad>', '<bos>'],
                        vectors={"word": [1, 2, 3]})

    def test_serialization(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        v = vocab.Vocab(c, min_freq=3, specials=['<pad>', '<bos>'])
        pickle_path = os.path.join(self.test_dir, "vocab.pkl")
        pickle.dump(v, open(pickle_path, "wb"))
        v_loaded = pickle.load(open(pickle_path, "rb"))
        assert v == v_loaded
