#!/usr/bin/env python3
"""Tests that requires external resources (Network access to fetch dataset)"""
import os
from collections import Counter

import torch
import torchtext.data

from .common.torchtext_test_case import TorchtextTestCase


class TestNestedField(TorchtextTestCase):
    def test_build_vocab(self):
        nesting_field = torchtext.legacy.data.Field(tokenize=list, init_token="<w>", eos_token="</w>")

        field = torchtext.legacy.data.NestedField(
            nesting_field, init_token='<s>', eos_token='</s>',
            include_lengths=True,
            pad_first=True)

        sources = [
            [['a'], ['s', 'e', 'n', 't', 'e', 'n', 'c', 'e'], ['o', 'f'], ['d', 'a', 't', 'a'], ['.']],
            [['y', 'e', 't'], ['a', 'n', 'o', 't', 'h', 'e', 'r']],
            [['o', 'n', 'e'], ['l', 'a', 's', 't'], ['s', 'e', 'n', 't']]
        ]

        field.build_vocab(
            sources, vectors='glove.6B.50d',
            unk_init=torch.nn.init.normal_, vectors_cache=".vector_cache")


class TestDataset(TorchtextTestCase):
    def test_csv_file_no_header_one_col_multiple_fields(self):
        self.write_test_ppid_dataset(data_format="csv")

        question_field = torchtext.legacy.data.Field(sequential=True)
        spacy_tok_question_field = torchtext.legacy.data.Field(sequential=True, tokenize="spacy")
        label_field = torchtext.legacy.data.Field(sequential=False)
        # Field name/value as nested tuples
        fields = [("ids", None),
                  (("q1", "q1_spacy"), (question_field, spacy_tok_question_field)),
                  (("q2", "q2_spacy"), (question_field, spacy_tok_question_field)),
                  ("label", label_field)]
        dataset = torchtext.legacy.data.TabularDataset(
            path=self.test_ppid_dataset_path, format="csv", fields=fields)
        expected_examples = [
            (["When", "do", "you", "use", "シ", "instead", "of", "し?"],
             ["When", "do", "you", "use", "シ", "instead", "of", "し", "?"],
             ["When", "do", "you", "use", "\"&\"",
              "instead", "of", "\"and\"?"],
             ["When", "do", "you", "use", "\"", "&", "\"",
              "instead", "of", "\"", "and", "\"", "?"], "0"),
            (["Where", "was", "Lincoln", "born?"],
             ["Where", "was", "Lincoln", "born", "?"],
             ["Which", "location", "was", "Abraham", "Lincoln", "born?"],
             ["Which", "location", "was", "Abraham", "Lincoln", "born", "?"],
             "1"),
            (["What", "is", "2+2"], ["What", "is", "2", "+", "2"],
             ["2+2=?"], ["2", "+", "2=", "?"], "1")]
        for i, example in enumerate(dataset):
            self.assertEqual(example.q1, expected_examples[i][0])
            self.assertEqual(example.q1_spacy, expected_examples[i][1])
            self.assertEqual(example.q2, expected_examples[i][2])
            self.assertEqual(example.q2_spacy, expected_examples[i][3])
            self.assertEqual(example.label, expected_examples[i][4])

        # 6 Fields including None for ids
        assert len(dataset.fields) == 6

    def test_json_dataset_one_key_multiple_fields(self):
        self.write_test_ppid_dataset(data_format="json")

        question_field = torchtext.legacy.data.Field(sequential=True)
        spacy_tok_question_field = torchtext.legacy.data.Field(sequential=True, tokenize="spacy")
        label_field = torchtext.legacy.data.Field(sequential=False)
        fields = {"question1": [("q1", question_field),
                                ("q1_spacy", spacy_tok_question_field)],
                  "question2": [("q2", question_field),
                                ("q2_spacy", spacy_tok_question_field)],
                  "label": ("label", label_field)}
        dataset = torchtext.legacy.data.TabularDataset(
            path=self.test_ppid_dataset_path, format="json", fields=fields)
        expected_examples = [
            (["When", "do", "you", "use", "シ", "instead", "of", "し?"],
             ["When", "do", "you", "use", "シ", "instead", "of", "し", "?"],
             ["When", "do", "you", "use", "\"&\"",
              "instead", "of", "\"and\"?"],
             ["When", "do", "you", "use", "\"", "&", "\"",
              "instead", "of", "\"", "and", "\"", "?"], "0"),
            (["Where", "was", "Lincoln", "born?"],
             ["Where", "was", "Lincoln", "born", "?"],
             ["Which", "location", "was", "Abraham", "Lincoln", "born?"],
             ["Which", "location", "was", "Abraham", "Lincoln", "born", "?"],
             "1"),
            (["What", "is", "2+2"], ["What", "is", "2", "+", "2"],
             ["2+2=?"], ["2", "+", "2=", "?"], "1")]
        for i, example in enumerate(dataset):
            self.assertEqual(example.q1, expected_examples[i][0])
            self.assertEqual(example.q1_spacy, expected_examples[i][1])
            self.assertEqual(example.q2, expected_examples[i][2])
            self.assertEqual(example.q2_spacy, expected_examples[i][3])
            self.assertEqual(example.label, expected_examples[i][4])


class TestDataUtils(TorchtextTestCase):
    TEST_STR = "A string, particularly one with slightly complex punctuation."

    def test_get_tokenizer_spacy(self):
        # Test SpaCy option, and verify it properly handles punctuation.
        assert torchtext.data.get_tokenizer("spacy")(str(self.TEST_STR)) == [
            "A", "string", ",", "particularly", "one", "with", "slightly",
            "complex", "punctuation", "."]

    def test_get_tokenizer_moses(self):
        # Test Moses option.
        # Note that internally, MosesTokenizer converts to unicode if applicable
        moses_tokenizer = torchtext.data.get_tokenizer("moses")
        assert moses_tokenizer(self.TEST_STR) == [
            "A", "string", ",", "particularly", "one", "with", "slightly",
            "complex", "punctuation", "."]

        # Nonbreaking prefixes should tokenize the final period.
        assert moses_tokenizer("abc def.") == ["abc", "def", "."]


class TestVocab(TorchtextTestCase):
    def test_vectors_get_vecs(self):
        vec = torchtext.vocab.GloVe(name='twitter.27B', dim='25')
        self.assertEqual(vec.vectors.shape[0], len(vec))

        tokens = ['chip', 'baby', 'Beautiful']
        token_vecs = vec.get_vecs_by_tokens(tokens)
        self.assertEqual(token_vecs.shape[0], len(tokens))
        self.assertEqual(token_vecs.shape[1], vec.dim)
        self.assertEqual(vec[tokens[0]], token_vecs[0])
        self.assertEqual(vec[tokens[1]], token_vecs[1])
        self.assertEqual(vec['<unk>'], token_vecs[2])

        token_one_vec = vec.get_vecs_by_tokens(tokens[0], lower_case_backup=True)
        self.assertEqual(token_one_vec.shape[0], vec.dim)
        self.assertEqual(vec[tokens[0].lower()], token_one_vec)

    def test_download_charngram_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching, then once more
        # to test string aliases.
        for i in range(3):
            if i == 2:
                vectors = "charngram.100d"
            else:
                vectors = torchtext.vocab.CharNGram()
            v = torchtext.vocab.Vocab(
                c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'], vectors=vectors)
            expected_itos = ['<unk>', '<pad>', '<bos>',
                             'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world']
            expected_stoi = {x: index for index, x in enumerate(expected_itos)}
            self.assertEqual(v.itos, expected_itos)
            self.assertEqual(dict(v.stoi), expected_stoi)
            vectors = v.vectors

            # The first 5 entries in each vector.
            expected_charngram = {
                'hello': [-0.44782442, -0.08937783, -0.34227219,
                          -0.16233221, -0.39343098],
                'world': [-0.29590717, -0.05275926, -0.37334684, 0.27117205, -0.3868292],
            }

            for word in expected_charngram:
                self.assertEqual(
                    vectors[v.stoi[word], :5], expected_charngram[word])

            self.assertEqual(vectors[v.stoi['<unk>']], torch.zeros(100))
            self.assertEqual(vectors[v.stoi['OOV token']], torch.zeros(100))

    def test_download_custom_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching.
        for _ in range(2):
            v = torchtext.vocab.Vocab(
                c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'],
                vectors=torchtext.vocab.Vectors(
                    'wiki.simple.vec',
                    url=torchtext.vocab.FastText.url_base.format('simple')
                )
            )

            self.assertEqual(v.itos, ['<unk>', '<pad>', '<bos>',
                                      'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world'])
            vectors = v.vectors

            # The first 5 entries in each vector.
            expected_fasttext_simple_en = {
                'hello': [0.39567, 0.21454, -0.035389, -0.24299, -0.095645],
                'world': [0.10444, -0.10858, 0.27212, 0.13299, -0.33165],
            }

            for word in expected_fasttext_simple_en:
                self.assertEqual(
                    vectors[v.stoi[word], :5], expected_fasttext_simple_en[word])

            self.assertEqual(vectors[v.stoi['<unk>']], torch.zeros(300))

    def test_download_fasttext_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching, then once more
        # to test string aliases.
        for i in range(3):
            if i == 2:
                vectors = "fasttext.simple.300d"
            else:
                vectors = torchtext.vocab.FastText(language='simple')

            v = torchtext.vocab.Vocab(
                c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'], vectors=vectors)

            expected_itos = ['<unk>', '<pad>', '<bos>',
                             'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world']
            expected_stoi = {x: index for index, x in enumerate(expected_itos)}
            self.assertEqual(v.itos, expected_itos)
            self.assertEqual(dict(v.stoi), expected_stoi)
            vectors = v.vectors

            # The first 5 entries in each vector.
            expected_fasttext_simple_en = {
                'hello': [0.39567, 0.21454, -0.035389, -0.24299, -0.095645],
                'world': [0.10444, -0.10858, 0.27212, 0.13299, -0.33165],
            }

            for word in expected_fasttext_simple_en:
                self.assertEqual(
                    vectors[v.stoi[word], :5], expected_fasttext_simple_en[word])

            self.assertEqual(vectors[v.stoi['<unk>']], torch.zeros(300))
            self.assertEqual(vectors[v.stoi['OOV token']], torch.zeros(300))

    def test_download_glove_vectors(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})

        # Build a vocab and get vectors twice to test caching, then once more
        # to test string aliases.
        for i in range(3):
            if i == 2:
                vectors = "glove.twitter.27B.25d"
            else:
                vectors = torchtext.vocab.GloVe(name='twitter.27B', dim='25')
            v = torchtext.vocab.Vocab(
                c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'], vectors=vectors)

            expected_itos = ['<unk>', '<pad>', '<bos>',
                             'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world']
            expected_stoi = {x: index for index, x in enumerate(expected_itos)}
            self.assertEqual(v.itos, expected_itos)
            self.assertEqual(dict(v.stoi), expected_stoi)

            vectors = v.vectors

            # The first 5 entries in each vector.
            expected_twitter = {
                'hello': [-0.77069, 0.12827, 0.33137, 0.0050893, -0.47605],
                'world': [0.10301, 0.095666, -0.14789, -0.22383, -0.14775],
            }

            for word in expected_twitter:
                self.assertEqual(
                    vectors[v.stoi[word], :5], expected_twitter[word])

            self.assertEqual(vectors[v.stoi['<unk>']], torch.zeros(25))
            self.assertEqual(vectors[v.stoi['OOV token']], torch.zeros(25))

    def test_extend(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        # Build a vocab and get vectors twice to test caching.
        for _ in range(2):
            f = torchtext.vocab.FastText(language='simple')
            v = torchtext.vocab.Vocab(
                c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'], vectors=f)
            n_vocab = len(v)
            v.extend(f)  # extend the vocab with the words contained in f.itos
            self.assertGreater(len(v), n_vocab)

            self.assertEqual(v.itos[:6], ['<unk>', '<pad>', '<bos>',
                                          'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world'])
            vectors = v.vectors

            # The first 5 entries in each vector.
            expected_fasttext_simple_en = {
                'hello': [0.39567, 0.21454, -0.035389, -0.24299, -0.095645],
                'world': [0.10444, -0.10858, 0.27212, 0.13299, -0.33165],
            }

            for word in expected_fasttext_simple_en:
                self.assertEqual(
                    vectors[v.stoi[word], :5], expected_fasttext_simple_en[word])

            self.assertEqual(vectors[v.stoi['<unk>']], torch.zeros(300))

    def test_vectors_custom_cache(self):
        c = Counter({'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2})
        vector_cache = os.path.join('/tmp', 'vector_cache')
        # Build a vocab and get vectors twice to test caching.
        for i in range(2):
            if i == 1:
                self.assertTrue(os.path.exists(vector_cache))

            v = torchtext.vocab.Vocab(
                c, min_freq=3, specials=['<unk>', '<pad>', '<bos>'],
                vectors=torchtext.vocab.Vectors(
                    'wiki.simple.vec', cache=vector_cache,
                    url=torchtext.vocab.FastText.url_base.format('simple'))
            )

            self.assertEqual(v.itos, ['<unk>', '<pad>', '<bos>',
                                      'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world'])
            vectors = v.vectors

            # The first 5 entries in each vector.
            expected_fasttext_simple_en = {
                'hello': [0.39567, 0.21454, -0.035389, -0.24299, -0.095645],
                'world': [0.10444, -0.10858, 0.27212, 0.13299, -0.33165],
            }

            for word in expected_fasttext_simple_en:
                self.assertEqual(
                    vectors[v.stoi[word], :5], expected_fasttext_simple_en[word])

            self.assertEqual(vectors[v.stoi['<unk>']], torch.zeros(300))
