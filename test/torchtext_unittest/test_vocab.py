# -*- coding: utf-8 -*-
import os
from collections import OrderedDict

import pytest
import torch
from torchtext.vocab import build_vocab_from_iterator, vocab
from torchtext_unittest.common.torchtext_test_case import TorchtextTestCase


class TestVocab(TorchtextTestCase):
    def tearDown(self) -> None:
        super().tearDown()
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()

    def test_vocab_membership(self) -> None:
        token_to_freq = {"<unk>": 2, "a": 2, "b": 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c, min_freq=2)

        self.assertTrue("<unk>" in v)
        self.assertTrue("a" in v)
        self.assertTrue("b" in v)
        self.assertFalse("c" in v)

    def test_vocab_get_item(self) -> None:
        token_to_freq = {"<unk>": 2, "a": 2, "b": 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c, min_freq=2)

        self.assertEqual(v["<unk>"], 0)
        self.assertEqual(v["a"], 1)
        self.assertEqual(v["b"], 2)

    def test_default_index(self) -> None:
        token_to_freq = {"<unk>": 2, "a": 2, "b": 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c, min_freq=2)

        self.assertTrue(v.get_default_index() is None)
        with self.assertRaises(RuntimeError):
            v["not in vocab"]

        v.set_default_index(0)
        self.assertEqual(v["not in vocab"], 0)

        v.set_default_index(None)
        with self.assertRaises(RuntimeError):
            v["not in vocab"]

    def test_default_index_jit(self) -> None:
        token_to_freq = {"<unk>": 2, "a": 2, "b": 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c, min_freq=2)
        v.set_default_index(0)
        v_jit = torch.jit.script(v)
        self.assertEqual(v_jit["not in vocab"], 0)

        v_jit.set_default_index(None)
        with self.assertRaises(RuntimeError):
            v_jit["not in vocab"]

    def test_vocab_insert_token(self) -> None:
        c = OrderedDict({"<unk>": 2, "a": 2})

        # add item to end
        v = vocab(c)
        v.insert_token("b", 2)

        expected_itos = ["<unk>", "a", "b"]
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(v.get_stoi()), expected_stoi)

        # add item to middle
        v = vocab(c)
        v.insert_token("b", 0)

        expected_itos = ["b", "<unk>", "a"]
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(v.get_stoi()), expected_stoi)

    def test_vocab_append_token(self) -> None:
        c = OrderedDict({"a": 2})
        v = vocab(c)
        v.append_token("b")

        expected_itos = ["a", "b"]
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(v.get_stoi()), expected_stoi)

        # token must not exist to be appended
        with self.assertRaises(RuntimeError):
            v.append_token("b")

    def test_vocab_len(self) -> None:
        token_to_freq = {"a": 2, "b": 2, "c": 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c)

        self.assertEqual(len(v), 3)

    def test_vocab_basic(self) -> None:
        token_to_freq = {"hello": 4, "world": 3, "ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T": 5, "freq_too_low": 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)

        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c, min_freq=3)

        expected_itos = ["ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T", "hello", "world"]
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(v.get_stoi()), expected_stoi)

    def test_vocab_jit(self) -> None:
        token_to_freq = {"hello": 4, "world": 3, "ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T": 5, "freq_too_low": 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)

        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c, min_freq=3)
        jit_v = torch.jit.script(v)

        expected_itos = ["ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T", "hello", "world"]
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        assert not v.is_jitable
        # Call the __prepare_scriptable__() func and convert the building block to the torbhind version
        # Not expect users to use the torchbind version on eager mode but still need a CI test here.
        assert v.__prepare_scriptable__().is_jitable

        self.assertEqual(jit_v.get_itos(), expected_itos)
        self.assertEqual(dict(jit_v.get_stoi()), expected_stoi)

    def test_vocab_forward(self) -> None:
        token_to_freq = {"a": 2, "b": 2, "c": 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)

        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c)
        jit_v = torch.jit.script(v)

        tokens = ["b", "a", "c"]
        expected_indices = [1, 0, 2]

        self.assertEqual(v(tokens), expected_indices)
        self.assertEqual(jit_v(tokens), expected_indices)

    def test_vocab_lookup_token(self) -> None:
        token_to_freq = {"a": 2, "b": 2, "c": 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c)

        self.assertEqual(v.lookup_token(1), "b")
        with self.assertRaises(RuntimeError):
            v.lookup_token(100)

    def test_vocab_lookup_tokens(self) -> None:
        token_to_freq = {"a": 2, "b": 2, "c": 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c)

        indices = [1, 0, 2]
        expected_tokens = ["b", "a", "c"]

        self.assertEqual(v.lookup_tokens(indices), expected_tokens)

    def test_vocab_lookup_indices(self) -> None:
        token_to_freq = {"a": 2, "b": 2, "c": 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c)

        tokens = ["b", "a", "c"]
        expected_indices = [1, 0, 2]

        self.assertEqual(v.lookup_indices(tokens), expected_indices)

    def test_vocab_load_and_save(self) -> None:
        token_to_freq = {"hello": 4, "world": 3, "ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T": 5, "freq_too_low": 2}
        sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)

        c = OrderedDict(sorted_by_freq_tuples)
        v = vocab(c, min_freq=3)
        v.set_default_index(0)

        expected_itos = ["ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T", "hello", "world"]
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}

        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(v.get_stoi()), expected_stoi)

        with self.subTest("pybind"):
            vocab_path = os.path.join(self.test_dir, "vocab_pybind.pt")
            torch.save(v, vocab_path)
            loaded_v = torch.load(vocab_path)
            self.assertEqual(v.get_itos(), expected_itos)
            self.assertEqual(dict(loaded_v.get_stoi()), expected_stoi)
            self.assertEqual(v["not in vocab"], 0)

        with self.subTest("torchscript"):
            vocab_path = os.path.join(self.test_dir, "vocab_torchscript.pt")
            # Call the __prepare_scriptable__() func and convert the building block to the torbhind version
            # Not expect users to use the torchbind version on eager mode but still need a CI test here.
            torch.save(v.__prepare_scriptable__(), vocab_path)
            loaded_v = torch.load(vocab_path)
            self.assertEqual(v.get_itos(), expected_itos)
            self.assertEqual(dict(loaded_v.get_stoi()), expected_stoi)
            self.assertEqual(v["not in vocab"], 0)

    def test_build_vocab_iterator(self) -> None:
        iterator = [
            [
                "hello",
                "hello",
                "hello",
                "freq_low",
                "hello",
                "world",
                "world",
                "world",
                "ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T",
                "ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T",
                "ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T",
                "freq_low",
                "ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T",
                "ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T",
            ]
        ]
        specials = ["<unk>", "<bos>", "<eos>", "pad"]
        v = build_vocab_from_iterator(iterator)
        expected_itos = ["ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T", "hello", "world", "freq_low"]
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}
        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(v.get_stoi()), expected_stoi)
        v = build_vocab_from_iterator(iterator, specials=specials)
        expected_itos = specials + ["ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T", "hello", "world", "freq_low"]
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}
        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(v.get_stoi()), expected_stoi)
        v = build_vocab_from_iterator(iterator, specials=specials, special_first=False)
        expected_itos = ["ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T", "hello", "world", "freq_low"] + specials
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}
        self.assertEqual(v.get_itos(), expected_itos)
        self.assertEqual(dict(v.get_stoi()), expected_stoi)

    def test_vocab_specials(self) -> None:
        token_to_freq = {"hello": 4, "world": 3, "ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T": 5, "freq_too_low": 2}
        sorted_by_freq_tuples = OrderedDict(sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True))
        specials = ["<unk>", "<bos>", "<eos>", "pad"]

        v1 = vocab(sorted_by_freq_tuples, min_freq=3, specials=specials)
        expected_itos = specials + ["ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T", "hello", "world"]
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}
        self.assertEqual(v1.get_itos(), expected_itos)
        self.assertEqual(dict(v1.get_stoi()), expected_stoi)

        v2 = vocab(sorted_by_freq_tuples, min_freq=3, specials=specials, special_first=False)
        expected_itos = ["ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T", "hello", "world"] + specials
        expected_stoi = {x: index for index, x in enumerate(expected_itos)}
        self.assertEqual(v2.get_itos(), expected_itos)
        self.assertEqual(dict(v2.get_stoi()), expected_stoi)

    def test_build_vocab_sorts_descending_frequency_then_lexigraphically(self) -> None:
        it = [["a", "b"], ["a", "b"]]
        vocab = build_vocab_from_iterator(it)
        self.assertEqual(vocab["a"], 0)
        self.assertEqual(vocab["b"], 1)

        it = [["a", "b"], ["b"]]
        vocab = build_vocab_from_iterator(it)
        self.assertEqual(vocab["b"], 0)
        self.assertEqual(vocab["a"], 1)

    def test_build_vocab_from_iterator_max_tokens(self) -> None:
        it = [["hello", "world"], ["hello"]]
        max_tokens = 1
        specials = ["<unk>", "<pad>"]
        self.assertLess(max_tokens, len(specials))
        with pytest.raises(AssertionError):
            build_vocab_from_iterator(it, specials=specials, max_tokens=max_tokens)

        max_tokens = 3
        vocab = build_vocab_from_iterator(it, specials=specials, special_first=True, max_tokens=max_tokens)
        self.assertEqual(vocab["<unk>"], 0)
        self.assertEqual(vocab["<pad>"], 1)
        self.assertEqual(vocab["hello"], 2)

        max_tokens = 3
        vocab = build_vocab_from_iterator(it, specials=specials, special_first=False, max_tokens=max_tokens)
        self.assertEqual(vocab["hello"], 0)
        self.assertEqual(vocab["<unk>"], 1)
        self.assertEqual(vocab["<pad>"], 2)
