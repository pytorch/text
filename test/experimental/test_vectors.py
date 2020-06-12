# -*- coding: utf-8 -*-
import torch

from test.common.assets import get_asset_path
from test.common.torchtext_test_case import TorchtextTestCase
from torchtext.experimental.vectors import (
    Vectors,
    vectors_from_csv_file
)


class TestVectors(TorchtextTestCase):

    def test_empty_vector(self):
        tokens = []
        vectors = []
        unk_tensor = torch.Tensor([0])

        vectorsObj = Vectors(tokens, vectors, unk_tensor)
        self.assertEqual(vectorsObj['not_in_it'].tolist(), unk_tensor.tolist())

    def test_empty_unk(self):
        tensorA = torch.Tensor([1, 0])
        expected_unk_tensor = torch.Tensor([0, 0])

        tokens = ['a']
        vectors = [tensorA]
        vectorsObj = Vectors(tokens, vectors)

        self.assertEqual(vectorsObj['not_in_it'].tolist(), expected_unk_tensor.tolist())

    def test_vectors_basic(self):
        tensorA = torch.Tensor([1, 0])
        tensorB = torch.Tensor([0, 1])
        unk_tensor = torch.Tensor([0, 0])

        tokens = ['a', 'b']
        vectors = [tensorA, tensorB]
        vectorsObj = Vectors(tokens, vectors, unk_tensor=unk_tensor)

        self.assertEqual(vectorsObj['a'].tolist(), tensorA.tolist())
        self.assertEqual(vectorsObj['b'].tolist(), tensorB.tolist())
        self.assertEqual(vectorsObj['not_in_it'].tolist(), unk_tensor.tolist())

    def test_errors(self):
        tokens = []
        vectors = []

        with self.assertRaises(ValueError):
            # Test proper error raised when passing in empty tokens and vectors and
            # not passing in a user defined unk_tensor
            Vectors(tokens, vectors)

        tensorA = torch.Tensor([1, 0, 0])
        tensorB = torch.Tensor([0, 1, 0])
        tokens = ['a', 'b', 'c']
        vectors = [tensorA, tensorB]

        with self.assertRaises(RuntimeError):
            # Test proper error raised when tokens and vectors have different sizes
            Vectors(tokens, vectors)

        tensorC = torch.Tensor([0, 0, 1])
        tokens = ['a', 'a', 'c']
        vectors = [tensorA, tensorB, tensorC]

        with self.assertRaises(RuntimeError):
            # Test proper error raised when tokens have duplicates
            # TODO (Nayef211): use self.assertRaisesRegex() to check
            # the key of the duplicate token in the error message
            Vectors(tokens, vectors)

    def test_vectors_from_file(self):
        asset_name = 'vectors_test.csv'
        asset_path = get_asset_path(asset_name)
        f = open(asset_path, 'r')
        vectorsObj = vectors_from_csv_file(f)

        expected_tensorA = torch.Tensor([1, 0, 0])
        expected_tensorB = torch.Tensor([0, 1, 0])
        expected_unk_tensor = torch.Tensor([0, 0, 0])

        self.assertEqual(vectorsObj['a'].tolist(), expected_tensorA.tolist())
        self.assertEqual(vectorsObj['b'].tolist(), expected_tensorB.tolist())
        self.assertEqual(vectorsObj['not_in_it'].tolist(), expected_unk_tensor.tolist())
