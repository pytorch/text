# -*- coding: utf-8 -*-
import os
import shutil
import tempfile

import torch

from test.common.assets import get_asset_path
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

    def test_vectors_call_method(self):
        tensorA = torch.tensor([1, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1], dtype=torch.float)

        unk_tensor = torch.tensor([0, 0], dtype=torch.float)
        tokens = ['a', 'b']
        vecs = torch.stack((tensorA, tensorB), 0)
        vectors_obj = vectors(tokens, vecs, unk_tensor=unk_tensor)

        tokens_to_lookup = ['a', 'b', 'c']
        expected_vectors = torch.stack((tensorA, tensorB, unk_tensor), 0)
        vectors_by_tokens = vectors_obj(tokens_to_lookup)

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

    def test_errors(self):
        tokens = []
        vecs = torch.empty(0, dtype=torch.float)

        with self.assertRaises(ValueError):
            # Test proper error raised when passing in empty tokens and vectors and
            # not passing in a user defined unk_tensor
            vectors(tokens, vecs)

        tensorA = torch.tensor([1, 0, 0], dtype=torch.float)
        tensorB = torch.tensor([0, 1, 0], dtype=torch.float)
        tokens = ['a', 'b', 'c']
        vecs = torch.stack((tensorA, tensorB,), 0)

        tensorC = torch.tensor([0, 0, 1], dtype=torch.float)
        tokens = ['a', 'a', 'c']
        vecs = torch.stack((tensorA, tensorB, tensorC), 0)

#        with self.assertRaises(RuntimeError):
#            # Test proper error raised when tokens have duplicates
#            # TODO (Nayef211): use self.assertRaisesRegex() to check
#            # the key of the duplicate token in the error message
#            vectors(tokens, vecs)

        tensorC = torch.tensor([0, 0, 1], dtype=torch.int8)
        tokens = ['a']
        vecs = tensorC.unsqueeze(0)

        with self.assertRaises(TypeError):
            # Test proper error raised when vector is not of type torch.float
            vectors(tokens, vecs)

        with tempfile.TemporaryDirectory() as dir_name:
            # Test proper error raised when incorrect filename or dim passed into GloVe
            asset_name = 'glove.6B.zip'
            asset_path = get_asset_path(asset_name)
            data_path = os.path.join(dir_name, asset_name)
            shutil.copy(asset_path, data_path)

            with self.assertRaises(ValueError):
                # incorrect name
                GloVe(name='UNK', dim=50, root=dir_name, validate_file=False)

            with self.assertRaises(ValueError):
                # incorrect dim
                GloVe(name='6B', dim=500, root=dir_name, validate_file=False)

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

    def test_glove(self):
        # copy the asset file into the expected download location
        # note that this is just a zip file with the first 100 entries of the GloVe 840B dataset
        asset_name = 'glove.840B.300d.zip'
        asset_path = get_asset_path(asset_name)

        with tempfile.TemporaryDirectory() as dir_name:
            data_path = os.path.join(dir_name, asset_name)
            shutil.copy(asset_path, data_path)
            vectors_obj = GloVe(root=dir_name, validate_file=False)
            jit_vectors_obj = torch.jit.script(vectors_obj.to_ivalue())

            # The first 3 entries in each vector.
            expected_glove = {
                'the': [0.27204, -0.06203, -0.1884],
                'people': [-0.19686, 0.11579, -0.41091],
            }

            for word in expected_glove.keys():
                self.assertEqual(vectors_obj[word][:3], expected_glove[word])
                self.assertEqual(jit_vectors_obj[word][:3], expected_glove[word])

    def test_glove_different_dims(self):
        # copy the asset file into the expected download location
        # note that this is just a zip file with 1 line txt files used to test that the
        # correct files are being loaded
        asset_name = 'glove.6B.zip'
        asset_path = get_asset_path(asset_name)

        with tempfile.TemporaryDirectory() as dir_name:
            data_path = os.path.join(dir_name, asset_name)
            shutil.copy(asset_path, data_path)

            glove_50d = GloVe(name='6B', dim=50, root=dir_name, validate_file=False)
            glove_100d = GloVe(name='6B', dim=100, root=dir_name, validate_file=False)
            glove_200d = GloVe(name='6B', dim=200, root=dir_name, validate_file=False)
            glove_300d = GloVe(name='6B', dim=300, root=dir_name, validate_file=False)
            vectors_objects = [glove_50d, glove_100d, glove_200d, glove_300d]

            # The first 3 entries in each vector.
            expected_glove_50d = {
                'the': [0.418, 0.24968, -0.41242],
            }
            expected_glove_100d = {
                'the': [-0.038194, -0.24487, 0.72812],
            }
            expected_glove_200d = {
                'the': [-0.071549, 0.093459, 0.023738],
            }
            expected_glove_300d = {
                'the': [0.04656, 0.21318, -0.0074364],
            }
            expected_gloves = [expected_glove_50d, expected_glove_100d, expected_glove_200d, expected_glove_300d]

            for vectors_obj, expected_glove in zip(vectors_objects, expected_gloves):
                for word in expected_glove.keys():
                    self.assertEqual(vectors_obj[word][:3], expected_glove[word])
