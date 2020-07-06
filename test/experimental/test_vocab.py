# -*- coding: utf-8 -*-
from collections import OrderedDict
import os
import torch

from test.common.torchtext_test_case import TorchtextTestCase
from torchtext.experimental.vocab import (
    Vocab
)


class TestVocab(TorchtextTestCase):
    def tearDown(self):
        super().tearDown()
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()

    def test_has_unk(self):
        c = OrderedDict({})
        v = Vocab(c)

        # check if unk is mapped to the first index
        self.assertEqual(v['not_in_it'], 0)
        self.assertEqual(v['<unk>'], 0)

    def test_new_unk(self):
        c = OrderedDict({})
        v = Vocab(c, unk_token="<new_unk>")

        # check if new_unk is mapped to the first index
        self.assertEqual(v['<new_unk>'], 0)
        self.assertEqual(v['not_in_it'], 0)

    def test_vocab_get_item(self):
        token_to_freq = {'a': 2, 'b': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = Vocab(c, specials=['<pad>', '<eos>'])

        self.assertEqual(v['<unk>'], 0)
        self.assertEqual(v['<pad>'], 1)
        self.assertEqual(v['<eos>'], 2)
        self.assertEqual(v['a'], 3)
        self.assertEqual(v['b'], 4)

    def test_vocab_set_item(self):
        c = OrderedDict({'a': 2})

        # add item to end
        v = Vocab(c, specials=['<pad>', '<eos>'])
        v['b'] = 4

        self.assertEqual(v['<unk>'], 0)
        self.assertEqual(v['<pad>'], 1)
        self.assertEqual(v['<eos>'], 2)
        self.assertEqual(v['a'], 3)
        self.assertEqual(v['b'], 4)

        # add item to middle
        v = Vocab(c, specials=['<pad>', '<eos>'], specials_first=False)
        v['b'] = 0

        self.assertEqual(v['b'], 0)
        self.assertEqual(v['a'], 1)
        self.assertEqual(v['<unk>'], 2)
        self.assertEqual(v['<pad>'], 3)
        self.assertEqual(v['<eos>'], 4)

    def test_vocab_add_token(self):
        c = OrderedDict({'a': 2})
        v = Vocab(c, specials=['<pad>', '<eos>'])
        v.add_token('b')

        self.assertEqual(len(v), 5)
        self.assertEqual(v['b'], 4)

    def test_vocab_len(self):
        token_to_freq = {'a': 2, 'b': 2, 'c': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = Vocab(c, specials=['<pad>', '<eos>'])

        self.assertEqual(len(v), 6)

    def test_vocab_basic(self):
        token_to_freq = {'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)

        c = OrderedDict(sorted_by_freq_tuples)
        v = Vocab(c, min_freq=3, specials=['<pad>', '<bos>'])

        expected_itos = ['<unk>', '<pad>', '<bos>',
                         'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world']
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(v.get_stoi()), expected_stoi)

    def test_vocab_jit(self):
        token_to_freq = {'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)

        c = OrderedDict(sorted_by_freq_tuples)
        v = Vocab(c, min_freq=3, specials=['<pad>', '<bos>'])
        jit_v = torch.jit.script(v)

        expected_itos = ['<unk>', '<pad>', '<bos>',
                         'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world']
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        self.assertEqual(jit_v.get_itos(), expected_itos)
        self.assertEqual(dict(jit_v.get_stoi()), expected_stoi)

    def test_vocab_specials_order(self):
        token_to_freq = {'a': 2, 'b': 2, 'c': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)

        # add specials into vocabulary at first
        v = Vocab(c, specials=['<pad>', '<eos>'])
        expected_itos = ['<unk>', '<pad>', '<eos>', 'a', 'b', 'c']
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        self.assertEqual(dict(v.get_stoi()), expected_stoi)

        # add specials into vocabulary at last
        v = Vocab(c, specials=['<pad>', '<eos>'], specials_first=False)
        expected_itos = ['a', 'b', 'c', '<unk>', '<pad>', '<eos>']
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        self.assertEqual(dict(v.get_stoi()), expected_stoi)

    def test_vocab_lookup_token(self):
        token_to_freq = {'a': 2, 'b': 2, 'c': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = Vocab(c, specials=['<pad>', '<eos>'], specials_first=False)

        self.assertEqual(v.lookup_token(0), 'a')

    def test_vocab_lookup_tokens(self):
        token_to_freq = {'a': 2, 'b': 2, 'c': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = Vocab(c, specials=['<pad>', '<eos>'], specials_first=False)

        indices = [1, 0, 2]
        expected_tokens = ['b', 'a', 'c']

        self.assertEqual(v.lookup_tokens(indices), expected_tokens)

    def test_vocab_lookup_indices(self):
        token_to_freq = {'a': 2, 'b': 2, 'c': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = Vocab(c, specials=['<pad>', '<eos>'], specials_first=False)

        tokens = ['b', 'a', 'c']
        expected_indices = [1, 0, 2]

        self.assertEqual(v.lookup_indices(tokens), expected_indices)

    def test_errors(self):
        token_to_freq = {'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)

        c = OrderedDict(sorted_by_freq_tuples)
        with self.assertRaises(ValueError):
            # Test proper error raised when setting unk token to None
            Vocab(c, min_freq=3, specials=['<pad>', '<bos>'], unk_token=None)

        with self.assertRaises(RuntimeError):
            # Test proper error raised when setting a token out of bounds
            v = Vocab(c, min_freq=3, specials=['<pad>', '<bos>'])
            v["new_token"] = 100

        with self.assertRaises(RuntimeError):
            # Test proper error raised when looking up a token out of bounds
            v = Vocab(c, min_freq=3, specials=['<pad>', '<bos>'])
            v["hello"] = 0

        with self.assertRaises(RuntimeError):
            # Test proper error raised when setting a token out of bounds
            v = Vocab(c, min_freq=3, specials=['<pad>', '<bos>'])
            v.lookup_token(100)

    def test_vocab_load_and_save(self):
        token_to_freq = {'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)

        c = OrderedDict(sorted_by_freq_tuples)
        v = Vocab(c, min_freq=3, specials=['<pad>', '<bos>'])

        expected_itos = ['<unk>', '<pad>', '<bos>',
                         'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T', 'hello', 'world']
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(v.get_stoi()), expected_stoi)

        vocab_path = os.path.join(self.test_dir, 'vocab.pt')
        torch.save(v, vocab_path)
        loaded_v = torch.load(vocab_path)

        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(loaded_v.get_stoi()), expected_stoi)
