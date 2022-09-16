# -*- coding: utf-8 -*-
import os
import platform
import unittest

import torch
from torchtext.prototype.vectors import build_vectors
from torchtext_unittest.common.torchtext_test_case import TorchtextTestCase


class TestVectors(TorchtextTestCase):
    def tearDown(self) -> None:
        super().tearDown()
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()

    def test_empty_vectors(self) -> None:
        tokens = []
        vecs = torch.empty(0, dtype=torch.float)
        unk_tensor = torch.tensor([0], dtype=torch.float)

        vectors_obj = build_vectors(tokens, vecs, unk_tensor)
        self.assertEqual(vectors_obj["not_in_it"], unk_tensor)

    def test_empty_unk(self) -> None:
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        expected_unk_tensor = torch.tensor([0, 0], dtype=torch.float)

        tokens = ["a"]
        vecs = tensorA.unsqueeze(0)
        vectors_obj = build_vectors(tokens, vecs)

        self.assertEqual(vectors_obj["not_in_it"], expected_unk_tensor)

    def test_vectors_basic(self) -> None:
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1], dtype=torch.float)

        unk_tensor = torch.tensor([0, 0], dtype=torch.float)
        tokens = ["a", "b"]
        vecs = torch.stack((tensorA, tensorB), 0)
        vectors_obj = build_vectors(tokens, vecs, unk_tensor=unk_tensor)

        self.assertEqual(vectors_obj["a"], tensorA)
        self.assertEqual(vectors_obj["b"], tensorB)
        self.assertEqual(vectors_obj["not_in_it"], unk_tensor)

    def test_vectors_jit(self) -> None:
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1], dtype=torch.float)

        unk_tensor = torch.tensor([0, 0], dtype=torch.float)
        tokens = ["a", "b"]
        vecs = torch.stack((tensorA, tensorB), 0)
        vectors_obj = build_vectors(tokens, vecs, unk_tensor=unk_tensor)
        jit_vectors_obj = torch.jit.script(vectors_obj)

        assert not vectors_obj.is_jitable
        # Call the __prepare_scriptable__() func and convert the building block to the torbhind version
        # Not expect users to use the torchbind version on eager mode but still need a CI test here.
        assert vectors_obj.__prepare_scriptable__().is_jitable

        self.assertEqual(vectors_obj["a"], jit_vectors_obj["a"])
        self.assertEqual(vectors_obj["b"], jit_vectors_obj["b"])
        self.assertEqual(vectors_obj["not_in_it"], jit_vectors_obj["not_in_it"])

    def test_vectors_forward(self) -> None:
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1], dtype=torch.float)

        unk_tensor = torch.tensor([0, 0], dtype=torch.float)
        tokens = ["a", "b"]
        vecs = torch.stack((tensorA, tensorB), 0)
        vectors_obj = build_vectors(tokens, vecs, unk_tensor=unk_tensor)
        jit_vectors_obj = torch.jit.script(vectors_obj)

        tokens_to_lookup = ["a", "b", "c"]
        expected_vectors = torch.stack((tensorA, tensorB, unk_tensor), 0)
        vectors_by_tokens = vectors_obj(tokens_to_lookup)
        jit_vectors_by_tokens = jit_vectors_obj(tokens_to_lookup)

        self.assertEqual(expected_vectors, vectors_by_tokens)
        self.assertEqual(expected_vectors, jit_vectors_by_tokens)

    def test_vectors_lookup_vectors(self) -> None:
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1], dtype=torch.float)

        unk_tensor = torch.tensor([0, 0], dtype=torch.float)
        tokens = ["a", "b"]
        vecs = torch.stack((tensorA, tensorB), 0)
        vectors_obj = build_vectors(tokens, vecs, unk_tensor=unk_tensor)

        tokens_to_lookup = ["a", "b", "c"]
        expected_vectors = torch.stack((tensorA, tensorB, unk_tensor), 0)
        vectors_by_tokens = vectors_obj.lookup_vectors(tokens_to_lookup)

        self.assertEqual(expected_vectors, vectors_by_tokens)

    def test_vectors_add_item(self) -> None:
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        unk_tensor = torch.tensor([0, 0], dtype=torch.float)

        tokens = ["a"]
        vecs = tensorA.unsqueeze(0)
        vectors_obj = build_vectors(tokens, vecs, unk_tensor=unk_tensor)

        tensorB = torch.tensor([0, 1], dtype=torch.float)
        vectors_obj["b"] = tensorB

        self.assertEqual(vectors_obj["a"], tensorA)
        self.assertEqual(vectors_obj["b"], tensorB)
        self.assertEqual(vectors_obj["not_in_it"], unk_tensor)

    def test_vectors_update(self) -> None:
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1], dtype=torch.float)
        tensorC = torch.tensor([1, 1], dtype=torch.float)

        expected_unk_tensor = torch.tensor([0, 0], dtype=torch.float)

        tokens = ["a", "b"]
        vecs = torch.stack((tensorA, tensorB), 0)
        vectors_obj = build_vectors(tokens, vecs)

        vectors_obj["b"] = tensorC

        self.assertEqual(vectors_obj["a"], tensorA)
        self.assertEqual(vectors_obj["b"], tensorC)
        self.assertEqual(vectors_obj["not_in_it"], expected_unk_tensor)

    def test_vectors_load_and_save(self) -> None:
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1], dtype=torch.float)
        expected_unk_tensor = torch.tensor([0, 0], dtype=torch.float)

        tokens = ["a", "b"]
        vecs = torch.stack((tensorA, tensorB), 0)
        vectors_obj = build_vectors(tokens, vecs)

        with self.subTest("pybind"):
            vector_path = os.path.join(self.test_dir, "vectors_pybind.pt")
            torch.save(vectors_obj, vector_path)
            loaded_vectors_obj = torch.load(vector_path)

            self.assertEqual(loaded_vectors_obj["a"], tensorA)
            self.assertEqual(loaded_vectors_obj["b"], tensorB)
            self.assertEqual(loaded_vectors_obj["not_in_it"], expected_unk_tensor)

        with self.subTest("torchscript"):
            vector_path = os.path.join(self.test_dir, "vectors_torchscript.pt")
            # Call the __prepare_scriptable__() func and convert the building block to the torbhind version
            # Not expect users to use the torchbind version on eager mode but still need a CI test here.
            torch.save(vectors_obj.__prepare_scriptable__(), vector_path)
            loaded_vectors_obj = torch.load(vector_path)

            self.assertEqual(loaded_vectors_obj["a"], tensorA)
            self.assertEqual(loaded_vectors_obj["b"], tensorB)
            self.assertEqual(loaded_vectors_obj["not_in_it"], expected_unk_tensor)

    # we separate out these errors because Windows runs into seg faults when propagating
    # exceptions from C++ using pybind11
    @unittest.skipIf(platform.system() == "Windows", "Test is known to fail on Windows.")
    def test_errors_vectors_cpp(self) -> None:
        tensorA = torch.tensor([1, 0, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1, 0], dtype=torch.float)
        tensorC = torch.tensor([0, 0, 1], dtype=torch.float)
        tokens = ["a", "a", "c"]
        vecs = torch.stack((tensorA, tensorB, tensorC), 0)

        with self.assertRaises(RuntimeError):
            # Test proper error raised when tokens have duplicates
            # TODO: use self.assertRaisesRegex() to check
            # the key of the duplicate token in the error message
            build_vectors(tokens, vecs)
