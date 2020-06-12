# -*- coding: utf-8 -*-
import torch

from test.common.assets import get_asset_path
from test.common.torchtext_test_case import TorchtextTestCase
from torchtext.experimental.vectors import (
    Vectors,
    vectors_from_csv_file
)


class TestVectors(TorchtextTestCase):

    def test_has_unk(self):
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

    def test_mismatch_vectors_tokens(self):
        tensorA = torch.Tensor([1, 0, 0])
        tensorB = torch.Tensor([0, 1, 0])
        expected_unk_tensor = torch.Tensor([0, 0, 0])

        tokens = ['a', 'b', 'c']
        vectors = [tensorA, tensorB]
        vectorsObj = Vectors(tokens, vectors)

        self.assertEqual(vectorsObj['a'].tolist(), tensorA.tolist())
        self.assertEqual(vectorsObj['b'].tolist(), tensorB.tolist())
        self.assertEqual(vectorsObj['c'].tolist(), expected_unk_tensor.tolist())

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
