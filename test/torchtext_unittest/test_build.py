#!/usr/bin/env python3
"""Tests that requires external resources (Network access to fetch dataset)"""
import os

import torch
import torchtext.data

from .common.torchtext_test_case import TorchtextTestCase, third_party_download


class TestDataUtils(TorchtextTestCase):
    TEST_STR = "A string, particularly one with slightly complex punctuation."

    def test_get_tokenizer_spacy(self) -> None:
        # Test SpaCy option, and verify it properly handles punctuation.
        assert torchtext.data.get_tokenizer("spacy", language="en_core_web_sm")(str(self.TEST_STR)) == [
            "A",
            "string",
            ",",
            "particularly",
            "one",
            "with",
            "slightly",
            "complex",
            "punctuation",
            ".",
        ]

    def test_get_tokenizer_moses(self) -> None:
        # Test Moses option.
        # Note that internally, MosesTokenizer converts to unicode if applicable
        moses_tokenizer = torchtext.data.get_tokenizer("moses")
        assert moses_tokenizer(self.TEST_STR) == [
            "A",
            "string",
            ",",
            "particularly",
            "one",
            "with",
            "slightly",
            "complex",
            "punctuation",
            ".",
        ]

        # Nonbreaking prefixes should tokenize the final period.
        assert moses_tokenizer("abc def.") == ["abc", "def", "."]


class TestVocab(TorchtextTestCase):
    def test_vectors_get_vecs(self) -> None:
        vec = torchtext.vocab.GloVe(name="twitter.27B", dim="25")
        self.assertEqual(vec.vectors.shape[0], len(vec))

        tokens = ["chip", "baby", "Beautiful"]
        token_vecs = vec.get_vecs_by_tokens(tokens)
        self.assertEqual(token_vecs.shape[0], len(tokens))
        self.assertEqual(token_vecs.shape[1], vec.dim)
        self.assertEqual(vec[tokens[0]], token_vecs[0])
        self.assertEqual(vec[tokens[1]], token_vecs[1])
        self.assertEqual(vec["<unk>"], token_vecs[2])

        token_one_vec = vec.get_vecs_by_tokens(tokens[0], lower_case_backup=True)
        self.assertEqual(token_one_vec.shape[0], vec.dim)
        self.assertEqual(vec[tokens[0].lower()], token_one_vec)

    @third_party_download
    def test_download_charngram_vectors(self) -> None:
        # Build a vocab and get vectors twice to test caching.
        for _ in range(2):
            vectors = torchtext.vocab.CharNGram()
            # The first 5 entries in each vector.
            expected_charngram = {
                "hello": [-0.44782442, -0.08937783, -0.34227219, -0.16233221, -0.39343098],
                "world": [-0.29590717, -0.05275926, -0.37334684, 0.27117205, -0.3868292],
            }

            for word in expected_charngram:
                self.assertEqual(vectors[word][0, :5], expected_charngram[word])

            self.assertEqual(vectors["<unk>"][0], torch.zeros(100))

            # The first 5 entries for `OOV token`
            expected_oov_token_charngram = [-0.1070, -0.2240, -0.3043, -0.1092, 0.0953]
            self.assertEqual(vectors["OOV token"][0, :5], expected_oov_token_charngram, atol=0, rtol=10e-4)

    def test_download_custom_vectors(self) -> None:
        # Build a vocab and get vectors twice to test caching.
        for _ in range(2):
            vectors = torchtext.vocab.Vectors("wiki.simple.vec", url=torchtext.vocab.FastText.url_base.format("simple"))

            # The first 5 entries in each vector.
            expected_fasttext_simple_en = {
                "hello": [0.39567, 0.21454, -0.035389, -0.24299, -0.095645],
                "world": [0.10444, -0.10858, 0.27212, 0.13299, -0.33165],
            }

            for word in expected_fasttext_simple_en:
                self.assertEqual(vectors[word][:5], expected_fasttext_simple_en[word])

            self.assertEqual(vectors["<unk>"], torch.zeros(300))

    def test_download_fasttext_vectors(self) -> None:
        # Build a vocab and get vectors twice to test caching.
        for _ in range(2):
            vectors = torchtext.vocab.FastText(language="simple")

            # The first 5 entries in each vector.
            expected_fasttext_simple_en = {
                "hello": [0.39567, 0.21454, -0.035389, -0.24299, -0.095645],
                "world": [0.10444, -0.10858, 0.27212, 0.13299, -0.33165],
            }

            for word in expected_fasttext_simple_en:
                self.assertEqual(vectors[word][:5], expected_fasttext_simple_en[word])

            self.assertEqual(vectors["<unk>"], torch.zeros(300))
            self.assertEqual(vectors["OOV token"], torch.zeros(300))

    def test_download_glove_vectors(self) -> None:
        # Build a vocab and get vectors twice to test caching.
        vectors = torchtext.vocab.GloVe(name="twitter.27B", dim="25")
        # The first 5 entries in each vector.
        expected_twitter = {
            "hello": [-0.77069, 0.12827, 0.33137, 0.0050893, -0.47605],
            "world": [0.10301, 0.095666, -0.14789, -0.22383, -0.14775],
        }

        for word in expected_twitter:
            self.assertEqual(vectors[word][:5], expected_twitter[word])

        self.assertEqual(vectors["<unk>"], torch.zeros(25))
        self.assertEqual(vectors["OOV token"], torch.zeros(25))

    def test_vectors_custom_cache(self) -> None:
        vector_cache = os.path.join("/tmp", "vector_cache")
        # Build a vocab and get vectors twice to test caching.
        for i in range(2):
            if i == 1:
                self.assertTrue(os.path.exists(vector_cache))

            vectors = torchtext.vocab.Vectors(
                "wiki.simple.vec", cache=vector_cache, url=torchtext.vocab.FastText.url_base.format("simple")
            )

            # The first 5 entries in each vector.
            expected_fasttext_simple_en = {
                "hello": [0.39567, 0.21454, -0.035389, -0.24299, -0.095645],
                "world": [0.10444, -0.10858, 0.27212, 0.13299, -0.33165],
            }

            for word in expected_fasttext_simple_en:
                self.assertEqual(vectors[word][:5], expected_fasttext_simple_en[word])

            self.assertEqual(vectors["<unk>"], torch.zeros(300))
