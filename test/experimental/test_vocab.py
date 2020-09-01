# -*- coding: utf-8 -*-
from collections import OrderedDict
import os
import platform
import torch
import unittest

from test.common.assets import get_asset_path
from test.common.torchtext_test_case import TorchtextTestCase
from torchtext.experimental.transforms import basic_english_normalize
from torchtext.experimental.vocab import (
    vocab,
    vocab_from_file,
    vocab_from_raw_text_file
)


class TestVocab(TorchtextTestCase):
    def tearDown(self):
        super().tearDown()
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()

    def test_has_unk(self):
        c = OrderedDict()
        v = vocab(c)

        # check if unk is mapped to the first index
        self.assertEqual(v['not_in_it'], 0)
        self.assertEqual(v['<unk>'], 0)

    def test_new_unk(self):
        c = OrderedDict()
        v = vocab(c, unk_token="<new_unk>")

        # check if new_unk is mapped to the first index
        self.assertEqual(v['<new_unk>'], 0)
        self.assertEqual(v['not_in_it'], 0)

    def test_vocab_get_item(self):
        token_to_freq = {'<unk>': 2, 'a': 2, 'b': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c, min_freq=2)

        self.assertEqual(v['<unk>'], 0)
        self.assertEqual(v['a'], 1)
        self.assertEqual(v['b'], 2)

    def test_vocab_insert_token(self):
        c = OrderedDict({'<unk>': 2, 'a': 2})

        # add item to end
        v = vocab(c)
        v.insert_token('b', 2)

        expected_itos = ['<unk>', 'a', 'b']
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(v.get_stoi()), expected_stoi)

        # add item to middle
        v = vocab(c)
        v.insert_token('b', 0)

        expected_itos = ['b', '<unk>', 'a']
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(v.get_stoi()), expected_stoi)

    def test_vocab_append_token(self):
        c = OrderedDict({'a': 2})
        v = vocab(c)
        v.append_token('b')

        expected_itos = ['<unk>', 'a', 'b']
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(v.get_stoi()), expected_stoi)

    def test_vocab_len(self):
        token_to_freq = {'a': 2, 'b': 2, 'c': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c)

        self.assertEqual(len(v), 4)

    def test_vocab_basic(self):
        token_to_freq = {'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)

        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c, min_freq=3)

        expected_itos = ['<unk>', 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world']
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(v.get_stoi()), expected_stoi)

    def test_vocab_jit(self):
        token_to_freq = {'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)

        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c, min_freq=3)
        jit_v = torch.jit.script(v.to_ivalue())

        expected_itos = ['<unk>', 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world']
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        assert not v.is_jitable
        assert v.to_ivalue().is_jitable

        self.assertEqual(jit_v.get_itos(), expected_itos)
        self.assertEqual(dict(jit_v.get_stoi()), expected_stoi)

    def test_vocab_forward(self):
        token_to_freq = {'a': 2, 'b': 2, 'c': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)

        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c)
        jit_v = torch.jit.script(v.to_ivalue())

        tokens = ['b', 'a', 'c']
        expected_indices = [2, 1, 3]

        self.assertEqual(v(tokens), expected_indices)
        self.assertEqual(jit_v(tokens), expected_indices)

    def test_vocab_lookup_token(self):
        token_to_freq = {'a': 2, 'b': 2, 'c': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c)

        self.assertEqual(v.lookup_token(1), 'a')

    def test_vocab_lookup_tokens(self):
        token_to_freq = {'a': 2, 'b': 2, 'c': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c)

        indices = [2, 1, 3]
        expected_tokens = ['b', 'a', 'c']

        self.assertEqual(v.lookup_tokens(indices), expected_tokens)

    def test_vocab_lookup_indices(self):
        token_to_freq = {'a': 2, 'b': 2, 'c': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c)

        tokens = ['b', 'a', 'c']
        expected_indices = [2, 1, 3]

        self.assertEqual(v.lookup_indices(tokens), expected_indices)

    def test_vocab_call_method(self):
        token_to_freq = {'a': 2, 'b': 2, 'c': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c)

        tokens = ['b', 'a', 'c']
        expected_indices = [2, 1, 3]

        self.assertEqual(v(tokens), expected_indices)

    # we seperate out these errors because Windows runs into seg faults when propagating
    # exceptions from C++ using pybind11
    @unittest.skipIf(platform.system() == "Windows", "Test is known to fail on Windows.")
    def test_errors_vocab_cpp(self):
        token_to_freq = {'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)

        with self.assertRaises(RuntimeError):
            # Test proper error raised when setting a token out of bounds
            v = vocab(c, min_freq=3)
            v.insert_token('new_token', 100)

        with self.assertRaises(RuntimeError):
            # Test proper error raised when looking up a token out of bounds
            v = vocab(c)
            v.lookup_token(100)

    def test_errors_vocab_python(self):
        token_to_freq = {'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)

        with self.assertRaises(ValueError):
            # Test proper error raised when setting unk token to None
            vocab(c, unk_token=None)

    def test_vocab_load_and_save(self):
        token_to_freq = {'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)

        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c, min_freq=3)

        expected_itos = ['<unk>', 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world']
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(v.get_stoi()), expected_stoi)

        vocab_path = os.path.join(self.test_dir, 'vocab.pt')
        torch.save(v.to_ivalue(), vocab_path)
        loaded_v = torch.load(vocab_path)

        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(loaded_v.get_stoi()), expected_stoi)

    def test_vocab_from_file(self):
        asset_name = 'vocab_test.txt'
        asset_path = get_asset_path(asset_name)
        f = open(asset_path, 'r')
        v = vocab_from_file(f, unk_token='<new_unk>')

        expected_itos = ['<new_unk>', 'a', 'b', 'c']
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(v.get_stoi()), expected_stoi)

    def test_vocab_from_raw_text_file(self):
        asset_name = 'vocab_raw_text_test.txt'
        asset_path = get_asset_path(asset_name)
        f = open(asset_path, 'r')

        tokenizer = basic_english_normalize()
        jit_tokenizer = torch.jit.script(tokenizer.to_ivalue())
        v = vocab_from_raw_text_file(f, jit_tokenizer, unk_token='<new_unk>')

        expected_itos = ['<new_unk>', 'fears', 'for', 't', 'n', 'pension', 'after',
                         'talks', 'unions', 'representing', 'workers', 'at', 'turner',
                         'newall', 'say', 'they', 'are', "'", 'disappointed', 'with',
                         'stricken', 'parent', 'firm', 'federal', 'mogul', '.']
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(v.get_stoi()), expected_stoi)
