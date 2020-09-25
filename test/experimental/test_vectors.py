# -*- coding: utf-8 -*-
import os
import platform
import shutil
import tempfile
import torch
import unittest
from ..common.assets import get_asset_path
from test.common.torchtext_test_case import TorchtextTestCase
from torchtext.experimental.vectors import (
    FastText,
    GloVe,
    vectors,
    vectors_from_file_object
)


class TestVectors(TorchtextTestCase):
    def tearDown(self):
        super().tearDown()
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()

    def test_empty_vectors(self):
        tokens = []
        vecs = torch.empty(0, dtype=torch.float)
        unk_tensor = torch.tensor([0], dtype=torch.float)

        vectors_obj = vectors(tokens, vecs, unk_tensor)
        self.assertEqual(vectors_obj['not_in_it'], unk_tensor)

    def test_empty_unk(self):
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        expected_unk_tensor = torch.tensor([0, 0], dtype=torch.float)

        tokens = ['a']
        vecs = tensorA.unsqueeze(0)
        vectors_obj = vectors(tokens, vecs)

        self.assertEqual(vectors_obj['not_in_it'], expected_unk_tensor)

    def test_vectors_basic(self):
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1], dtype=torch.float)

        unk_tensor = torch.tensor([0, 0], dtype=torch.float)
        tokens = ['a', 'b']
        vecs = torch.stack((tensorA, tensorB), 0)
        vectors_obj = vectors(tokens, vecs, unk_tensor=unk_tensor)

        self.assertEqual(vectors_obj['a'], tensorA)
        self.assertEqual(vectors_obj['b'], tensorB)
        self.assertEqual(vectors_obj['not_in_it'], unk_tensor)

    def test_vectors_jit(self):
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1], dtype=torch.float)

        unk_tensor = torch.tensor([0, 0], dtype=torch.float)
        tokens = ['a', 'b']
        vecs = torch.stack((tensorA, tensorB), 0)
        vectors_obj = vectors(tokens, vecs, unk_tensor=unk_tensor)
        jit_vectors_obj = torch.jit.script(vectors_obj.to_ivalue())

        assert not vectors_obj.is_jitable
        assert vectors_obj.to_ivalue().is_jitable

        self.assertEqual(vectors_obj['a'], jit_vectors_obj['a'])
        self.assertEqual(vectors_obj['b'], jit_vectors_obj['b'])
        self.assertEqual(vectors_obj['not_in_it'], jit_vectors_obj['not_in_it'])

    def test_vectors_forward(self):
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1], dtype=torch.float)

        unk_tensor = torch.tensor([0, 0], dtype=torch.float)
        tokens = ['a', 'b']
        vecs = torch.stack((tensorA, tensorB), 0)
        vectors_obj = vectors(tokens, vecs, unk_tensor=unk_tensor)
        jit_vectors_obj = torch.jit.script(vectors_obj.to_ivalue())

        tokens_to_lookup = [['a', 'b', 'c']]
        expected_vectors = [torch.stack((tensorA, tensorB, unk_tensor), 0)]
        vectors_by_tokens = vectors_obj(tokens_to_lookup)
        jit_vectors_by_tokens = jit_vectors_obj(tokens_to_lookup)

        self.assertEqual(expected_vectors, vectors_by_tokens)
        self.assertEqual(expected_vectors, jit_vectors_by_tokens)

    def test_vectors_lookup_vectors(self):
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1], dtype=torch.float)

        unk_tensor = torch.tensor([0, 0], dtype=torch.float)
        tokens = ['a', 'b']
        vecs = torch.stack((tensorA, tensorB), 0)
        vectors_obj = vectors(tokens, vecs, unk_tensor=unk_tensor)

        tokens_to_lookup = ['a', 'b', 'c']
        expected_vectors = torch.stack((tensorA, tensorB, unk_tensor), 0)
        vectors_by_tokens = vectors_obj.lookup_vectors(tokens_to_lookup)

        self.assertEqual(expected_vectors, vectors_by_tokens)

    def test_vectors_add_item(self):
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        unk_tensor = torch.tensor([0, 0], dtype=torch.float)

        tokens = ['a']
        vecs = tensorA.unsqueeze(0)
        vectors_obj = vectors(tokens, vecs, unk_tensor=unk_tensor)

        tensorB = torch.tensor([0, 1], dtype=torch.float)
        vectors_obj['b'] = tensorB

        self.assertEqual(vectors_obj['a'], tensorA)
        self.assertEqual(vectors_obj['b'], tensorB)
        self.assertEqual(vectors_obj['not_in_it'], unk_tensor)

    def test_vectors_load_and_save(self):
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1], dtype=torch.float)
        expected_unk_tensor = torch.tensor([0, 0], dtype=torch.float)

        tokens = ['a', 'b']
        vecs = torch.stack((tensorA, tensorB), 0)
        vectors_obj = vectors(tokens, vecs)

        tensorC = torch.tensor([1, 1], dtype=torch.float)
        vectors_obj['b'] = tensorC

        vector_path = os.path.join(self.test_dir, 'vectors.pt')
        torch.save(vectors_obj.to_ivalue(), vector_path)
        loaded_vectors_obj = torch.load(vector_path)

        self.assertEqual(loaded_vectors_obj['a'], tensorA)
        self.assertEqual(loaded_vectors_obj['b'], tensorC)
        self.assertEqual(loaded_vectors_obj['not_in_it'], expected_unk_tensor)

    # we separate out these errors because Windows runs into seg faults when propagating
    # exceptions from C++ using pybind11
    @unittest.skipIf(platform.system() == "Windows", "Test is known to fail on Windows.")
    def test_errors_vectors_cpp(self):
        tensorA = torch.tensor([1, 0, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1, 0], dtype=torch.float)
        tensorC = torch.tensor([0, 0, 1], dtype=torch.float)
        tokens = ['a', 'a', 'c']
        vecs = torch.stack((tensorA, tensorB, tensorC), 0)

        with self.assertRaises(RuntimeError):
            # Test proper error raised when tokens have duplicates
            # TODO: use self.assertRaisesRegex() to check
            # the key of the duplicate token in the error message
            vectors(tokens, vecs)

    def test_vectors_from_file(self):
        asset_name = 'vectors_test.csv'
        asset_path = get_asset_path(asset_name)
        f = open(asset_path, 'r')
        vectors_obj = vectors_from_file_object(f)

        expected_tensorA = torch.tensor([1, 0, 0], dtype=torch.float)
        expected_tensorB = torch.tensor([0, 1, 0], dtype=torch.float)
        expected_unk_tensor = torch.tensor([0, 0, 0], dtype=torch.float)

        self.assertEqual(vectors_obj['a'], expected_tensorA)
        self.assertEqual(vectors_obj['b'], expected_tensorB)
        self.assertEqual(vectors_obj['not_in_it'], expected_unk_tensor)

    def test_fast_text(self):
        # copy the asset file into the expected download location
        # note that this is just a file with the first 100 entries of the FastText english dataset
        asset_name = 'wiki.en.vec'
        asset_path = get_asset_path(asset_name)

        with tempfile.TemporaryDirectory() as dir_name:
            data_path = os.path.join(dir_name, asset_name)
            shutil.copy(asset_path, data_path)
            vectors_obj = FastText(root=dir_name, validate_file=False)
            jit_vectors_obj = torch.jit.script(vectors_obj.to_ivalue())

            # The first 3 entries in each vector.
            expected_fasttext_simple_en = {
                'the': [-0.065334, -0.093031, -0.017571],
                'world': [-0.32423, -0.098845, -0.0073467],
            }

            for word in expected_fasttext_simple_en.keys():
                self.assertEqual(vectors_obj[word][:3], expected_fasttext_simple_en[word])
                self.assertEqual(jit_vectors_obj[word][:3], expected_fasttext_simple_en[word])
