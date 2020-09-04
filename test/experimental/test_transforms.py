import torch
from test.common.torchtext_test_case import TorchtextTestCase
from test.common.assets import get_asset_path
from torchtext.experimental.transforms import (
    VectorTransform,
    VocabTransform,
)
from torchtext.experimental.vocab import vocab_from_file
from torchtext.experimental.vectors import FastText
import shutil
import tempfile
import os


class TestTransforms(TorchtextTestCase):
    def test_vocab_transform(self):
        asset_name = 'vocab_test2.txt'
        asset_path = get_asset_path(asset_name)
        with open(asset_path, 'r') as f:
            vocab_transform = VocabTransform(vocab_from_file(f))
            self.assertEqual(vocab_transform([['of', 'that', 'new'], ['of', 'that', 'new', 'that']]),
                             [[21, 26, 20], [21, 26, 20, 26]])
            jit_vocab_transform = torch.jit.script(vocab_transform.to_ivalue())
            self.assertEqual(jit_vocab_transform([['of', 'that', 'new'], ['of', 'that', 'new', 'that']]),
                             [[21, 26, 20], [21, 26, 20, 26]])

    def test_vector_transform(self):
        asset_name = 'wiki.en.vec'
        asset_path = get_asset_path(asset_name)

        with tempfile.TemporaryDirectory() as dir_name:
            data_path = os.path.join(dir_name, asset_name)
            shutil.copy(asset_path, data_path)
            vector_transform = VectorTransform(FastText(root=dir_name, validate_file=False))
            jit_vector_transform = torch.jit.script(vector_transform.to_ivalue())
            # The first 3 entries in each vector.
            expected_fasttext_simple_en = torch.tensor([[-0.065334, -0.093031, -0.017571],
                                                        [-0.32423, -0.098845, -0.0073467]])
            self.assertEqual(vector_transform([['the', 'world']])[0][:, 0:3], expected_fasttext_simple_en)
            self.assertEqual(jit_vector_transform([['the', 'world']])[0][:, 0:3], expected_fasttext_simple_en)
