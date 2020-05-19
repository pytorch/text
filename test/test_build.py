#!/usr/bin/env python3
"""Tests that requires external resources (Network access to fetch dataset)"""

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
