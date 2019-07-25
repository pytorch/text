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
        v = vocab.Vocab(c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'])

        expected_itos = ['<unk>', '<pad>', '<bos>',
                         'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world']
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}
        self.assertEqual(v.itos, expected_itos)
        self.assertEqual(dict(v.stoi), expected_stoi)

    def test_vocab_specials_first(self):
        c = Counter("a a b b c c".split())

        # add specials into vocabulary at first
        v = vocab.Vocab(c, max_size=2, specials=['<pad>', '<eos>'])
        expected_itos = ['<pad>', '<eos>', 'a', 'b']
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}
        self.assertEqual(v.itos, expected_itos)
        self.assertEqual(dict(v.stoi), expected_stoi)

        # add specials into vocabulary at last
        v = vocab.Vocab(c, max_size=2, specials=['<pad>', '<eos>'], specials_first=False)
        expected_itos = ['a', 'b', '<pad>', '<eos>']
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}
        self.assertEqual(v.itos, expected_itos)
        self.assertEqual(dict(v.stoi), expected_stoi)

    def test_vocab_without_unk(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        oov_word = 'OOVWORD'
        self.assertNotIn(oov_word, c)

        # tests for specials_first=True
        v_first = vocab.Vocab(c, min_freq=3, specials=['<pad>'], specials_first=True)
        expected_itos_first = ['<pad>', 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world']
        expected_stoi_first = {x: index for index, x in enumerate(expected_itos_first)}
        self.assertEqual(v_first.itos, expected_itos_first)
        self.assertEqual(dict(v_first.stoi), expected_stoi_first)
        self.assertNotIn(oov_word, v_first.itos)
        self.assertNotIn(oov_word, v_first.stoi)

        # tests for specials_first=False
        v_last = vocab.Vocab(c, min_freq=3, specials=['<pad>'], specials_first=False)
        expected_itos_last = ['ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world', '<pad>']
        expected_stoi_last = {x: index for index, x in enumerate(expected_itos_last)}
        self.assertEqual(v_last.itos, expected_itos_last)
        self.assertEqual(dict(v_last.stoi), expected_stoi_last)
        self.assertNotIn(oov_word, v_last.itos)
        self.assertNotIn(oov_word, v_last.stoi)

        # check if pad is mapped to the first index
        self.assertEqual(v_first.stoi['<pad>'], 0)
        # check if pad is mapped to the last index
        self.assertEqual(v_last.stoi['<pad>'], max(v_last.stoi.values()))

        # check if an oovword is not in vocab and a default unk_id is not assigned to it
        self.assertRaises(KeyError, v_first.stoi.__getitem__, oov_word)
        self.assertRaises(KeyError, v_last.stoi.__getitem__, oov_word)

    def test_vocab_set_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5,
                     'ｔｅｓｔ': 4, 'freq_too_low': 2})
        v = vocab.Vocab(c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'])
        stoi = {"hello": 0, "world": 1, "ｔｅｓｔ": 2}
        vectors = torch.FloatTensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        dim = 2
        v.set_vectors(stoi, vectors, dim)
        expected_vectors = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                     [0.0, 0.0], [0.1, 0.2], [0.5, 0.6],
                                     [0.3, 0.4]])
        assert_allclose(v.vectors.numpy(), expected_vectors)

    @slow
    def test_vocab_download_fasttext_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching, then once more
        # to test string aliases.
        for i in range(3):
            if i == 2:
                vectors = str("fasttext.simple.300d")  # must handle str on Py2
            else:
                vectors = FastText(language='simple')

            v = vocab.Vocab(c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'],
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

    @slow
    def test_vocab_extend(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching.
        for i in range(2):
            f = FastText(language='simple')
            v = vocab.Vocab(c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'],
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

    @slow
    def test_vocab_download_custom_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching.
        for i in range(2):
            v = vocab.Vocab(c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'],
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
    def test_vocab_vectors_custom_cache(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        vector_cache = os.path.join('/tmp', 'vector_cache')
        # Build a vocab and get vectors twice to test caching.
        for i in range(2):
            if i == 1:
                self.assertTrue(os.path.exists(vector_cache))

            v = vocab.Vocab(c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'],
                            vectors=Vectors('wiki.simple.vec', cache=vector_cache,
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
            vec_file = os.path.join(vector_cache, "wiki.simple.vec")
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
            v = vocab.Vocab(c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'],
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
            v = vocab.Vocab(c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'],
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
            vocab.Vocab(c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'],
                        vectors=["fasttext.english.300d"])
            vocab.Vocab(c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'],
                        vectors="fasttext.english.300d")
        with self.assertRaises(ValueError):
            # Test proper error is raised when vectors argument is
            # non-string or non-Vectors
            vocab.Vocab(c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'],
                        vectors={"word": [1, 2, 3]})

    def test_serialization(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        v = vocab.Vocab(c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'])
        pickle_path = os.path.join(self.test_dir, "vocab.pkl")
        pickle.dump(v, open(pickle_path, "wb"))
        v_loaded = pickle.load(open(pickle_path, "rb"))
        assert v == v_loaded

    @slow
    def test_vectors_get_vecs(self):
        vec = GloVe(name='twitter.27B', dim='25')
        self.assertEqual(vec.vectors.shape[0], len(vec))

        tokens = ['chip', 'baby', 'Beautiful']
        token_vecs = vec.get_vecs_by_tokens(tokens).numpy()
        self.assertEqual(token_vecs.shape[0], len(tokens))
        self.assertEqual(token_vecs.shape[1], vec.dim)
        assert_allclose(vec[tokens[0]].numpy(), token_vecs[0])
        assert_allclose(vec[tokens[1]].numpy(), token_vecs[1])
        assert_allclose(vec['<unk>'].numpy(), token_vecs[2])

        token_one_vec = vec.get_vecs_by_tokens(tokens[0], lower_case_backup=True).numpy()
        self.assertEqual(token_one_vec.shape[0], vec.dim)
        assert_allclose(vec[tokens[0].lower()].numpy(), token_one_vec)

        # Delete the vectors after we're done to save disk space on CI
        if os.environ.get("TRAVIS") == "true":
            zip_file = os.path.join(self.project_root, ".vector_cache",
                                    "glove.6B.zip")
            conditional_remove(zip_file)
            for dim in ["50", "100", "200", "300"]:
                conditional_remove(os.path.join(self.project_root, ".vector_cache",
                                                "glove.6B.{}d.txt".format(dim)))
