# -*- coding: utf-8 -*-
import os
import shutil
import tempfile

import numpy as np
import torch

from test.common.assets import get_asset_path
from test.common.torchtext_test_case import TorchtextTestCase
from torchtext.experimental.vectors import (
    fast_text,
    glo_ve,
    Vectors,
    vectors_from_file_object
)


class TestVectors(TorchtextTestCase):

    def test_empty_vectors(self):
        tokens = []
        vectors = []
        unk_tensor = torch.tensor([0], dtype=torch.float)

        vectors_obj = Vectors(tokens, vectors, unk_tensor)
        torch.testing.assert_allclose(vectors_obj['not_in_it'], unk_tensor)

    def test_empty_unk(self):
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        expected_unk_tensor = torch.tensor([0, 0], dtype=torch.float)

        tokens = ['a']
        vectors = [tensorA]
        vectors_obj = Vectors(tokens, vectors)

        torch.testing.assert_allclose(vectors_obj['not_in_it'], expected_unk_tensor)

    def test_vectors_basic(self):
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1], dtype=torch.float)

        unk_tensor = torch.tensor([0, 0], dtype=torch.float)
        tokens = ['a', 'b']
        vectors = [tensorA, tensorB]
        vectors_obj = Vectors(tokens, vectors, unk_tensor=unk_tensor)

        torch.testing.assert_allclose(vectors_obj['a'], tensorA)
        torch.testing.assert_allclose(vectors_obj['b'], tensorB)
        torch.testing.assert_allclose(vectors_obj['not_in_it'], unk_tensor)

    def test_vectors_jit(self):
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1], dtype=torch.float)

        unk_tensor = torch.tensor([0, 0], dtype=torch.float)
        tokens = ['a', 'b']
        vectors = [tensorA, tensorB]
        vectors_obj = Vectors(tokens, vectors, unk_tensor=unk_tensor)
        jit_vectors_obj = torch.jit.script(vectors_obj)

        torch.testing.assert_allclose(vectors_obj['a'], jit_vectors_obj['a'])
        torch.testing.assert_allclose(vectors_obj['b'], jit_vectors_obj['b'])
        torch.testing.assert_allclose(vectors_obj['not_in_it'], jit_vectors_obj['not_in_it'])

    def test_vectors_add_item(self):
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        unk_tensor = torch.tensor([0, 0], dtype=torch.float)

        tokens = ['a']
        vectors = [tensorA]
        vectors_obj = Vectors(tokens, vectors, unk_tensor=unk_tensor)

        tensorB = torch.tensor([0., 1])
        vectors_obj['b'] = tensorB

        torch.testing.assert_allclose(vectors_obj['a'], tensorA)
        torch.testing.assert_allclose(vectors_obj['b'], tensorB)
        torch.testing.assert_allclose(vectors_obj['not_in_it'], unk_tensor)

    def test_vectors_load_and_save(self):
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        expected_unk_tensor = torch.tensor([0, 0], dtype=torch.float)

        tokens = ['a']
        vectors = [tensorA]
        vectors_obj = Vectors(tokens, vectors)

        vector_path = os.path.join(self.test_dir, 'vectors.pt')
        torch.save(vectors_obj, vector_path)
        loaded_vectors_obj = torch.load(vector_path)

        torch.testing.assert_allclose(loaded_vectors_obj['a'], tensorA)
        torch.testing.assert_allclose(loaded_vectors_obj['not_in_it'], expected_unk_tensor)

    def test_errors(self):
        tokens = []
        vectors = []

        with self.assertRaises(ValueError):
            # Test proper error raised when passing in empty tokens and vectors and
            # not passing in a user defined unk_tensor
            Vectors(tokens, vectors)

        tensorA = torch.tensor([1, 0, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1, 0], dtype=torch.float)
        tokens = ['a', 'b', 'c']
        vectors = [tensorA, tensorB]

        with self.assertRaises(RuntimeError):
            # Test proper error raised when tokens and vectors have different sizes
            Vectors(tokens, vectors)

        tensorC = torch.tensor([0, 0, 1], dtype=torch.float)
        tokens = ['a', 'a', 'c']
        vectors = [tensorA, tensorB, tensorC]

        with self.assertRaises(RuntimeError):
            # Test proper error raised when tokens have duplicates
            # TODO (Nayef211): use self.assertRaisesRegex() to check
            # the key of the duplicate token in the error message
            Vectors(tokens, vectors)

        tensorC = torch.tensor([0, 0, 1], dtype=torch.int8)
        vectors = [tensorA, tensorB, tensorC]

        with self.assertRaises(TypeError):
            # Test proper error raised when vectors are not of type torch.float
            Vectors(tokens, vectors)

    def test_vectors_from_file(self):
        asset_name = 'vectors_test.csv'
        asset_path = get_asset_path(asset_name)
        f = open(asset_path, 'r')
        vectors_obj = vectors_from_file_object(f)

        expected_tensorA = torch.tensor([1, 0, 0], dtype=torch.float)
        expected_tensorB = torch.tensor([0, 1, 0], dtype=torch.float)
        expected_unk_tensor = torch.tensor([0, 0, 0], dtype=torch.float)

        torch.testing.assert_allclose(vectors_obj['a'], expected_tensorA)
        torch.testing.assert_allclose(vectors_obj['b'], expected_tensorB)
        torch.testing.assert_allclose(vectors_obj['not_in_it'], expected_unk_tensor)

    def test_fast_text(self):
        # copy the asset file into the expected download location
        asset_name = 'wiki.en.vec'
        asset_path = get_asset_path(asset_name)

        with tempfile.TemporaryDirectory() as dir_name:
            dir_name = ".tmp"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            data_path = os.path.join(dir_name, asset_name)
            shutil.copy(asset_path, data_path)

            print(asset_path)
            print(data_path)
            vectors_obj = fast_text(root=dir_name)

            # The first 3 entries in each vector.
            expected_fasttext_simple_en = {
                'the': [-0.065334, -0.093031, -0.017571],
                'world': [-0.32423, -0.098845, -0.0073467],
            }

            for word in expected_fasttext_simple_en.keys():
                torch.testing.assert_allclose(vectors_obj[word][:3], expected_fasttext_simple_en[word])

            def test_glo_ve(self):
                vectors_obj = glo_ve()

            # # The first 3 entries in each vector.
            # expected_twitter = {
            #     'hello': [-0.15945, -0.18259, 0.033443],
            #     'world': [-0.32423, -0.098845, -0.0073467],
            # }

            # for word in expected_twitter.keys():
            #     print(vectors_obj[word][:3])
            #     torch.testing.assert_allclose(vectors_obj[word][:3].tolist(), expected_twitter[word])

            # print(vectors_obj['hello'][:3])
            # print(vectors_obj['world'][:3])
