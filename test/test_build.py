#!/usr/bin/env python3
"""Tests that requires external resources (Network access to fetch dataset)"""
from collections import Counter

import numpy as np
import torch
import torchtext.data

from .common.torchtext_test_case import TorchtextTestCase


class TestNestedField(TorchtextTestCase):
    def test_build_vocab(self):
        nesting_field = torchtext.data.Field(tokenize=list, init_token="<w>", eos_token="</w>")

        field = torchtext.data.NestedField(
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

        question_field = torchtext.data.Field(sequential=True)
        spacy_tok_question_field = torchtext.data.Field(sequential=True, tokenize="spacy")
        label_field = torchtext.data.Field(sequential=False)
        # Field name/value as nested tuples
        fields = [("ids", None),
                  (("q1", "q1_spacy"), (question_field, spacy_tok_question_field)),
                  (("q2", "q2_spacy"), (question_field, spacy_tok_question_field)),
                  ("label", label_field)]
        dataset = torchtext.data.TabularDataset(
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

        question_field = torchtext.data.Field(sequential=True)
        spacy_tok_question_field = torchtext.data.Field(sequential=True, tokenize="spacy")
        label_field = torchtext.data.Field(sequential=False)
        fields = {"question1": [("q1", question_field),
                                ("q1_spacy", spacy_tok_question_field)],
                  "question2": [("q2", question_field),
                                ("q2_spacy", spacy_tok_question_field)],
                  "label": ("label", label_field)}
        dataset = torchtext.data.TabularDataset(
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


class TestVocab(TorchtextTestCase):
    def test_vectors_get_vecs(self):
        vec = torchtext.vocab.GloVe(name='twitter.27B', dim='25')
        self.assertEqual(vec.vectors.shape[0], len(vec))

        tokens = ['chip', 'baby', 'Beautiful']
        token_vecs = vec.get_vecs_by_tokens(tokens).numpy()
        self.assertEqual(token_vecs.shape[0], len(tokens))
        self.assertEqual(token_vecs.shape[1], vec.dim)
        torch.testing.assert_allclose(vec[tokens[0]].numpy(), token_vecs[0])
        torch.testing.assert_allclose(vec[tokens[1]].numpy(), token_vecs[1])
        torch.testing.assert_allclose(vec['<unk>'].numpy(), token_vecs[2])

        token_one_vec = vec.get_vecs_by_tokens(tokens[0], lower_case_backup=True).numpy()
        self.assertEqual(token_one_vec.shape[0], vec.dim)
        torch.testing.assert_allclose(vec[tokens[0].lower()].numpy(), token_one_vec)


    def test_vocab_download_charngram_vectors(self):
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
            vectors = v.vectors.numpy()

            # The first 5 entries in each vector.
            expected_charngram = {
                'hello': [-0.44782442, -0.08937783, -0.34227219,
                          -0.16233221, -0.39343098],
                'world': [-0.29590717, -0.05275926, -0.37334684, 0.27117205, -0.3868292],
            }

            for word in expected_charngram:
                torch.testing.assert_allclose(
                    vectors[v.stoi[word], :5], expected_charngram[word])

            torch.testing.assert_allclose(vectors[v.stoi['<unk>']], np.zeros(100))
            torch.testing.assert_allclose(vectors[v.stoi['OOV token']], np.zeros(100))
