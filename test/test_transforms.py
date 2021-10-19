import torch
from torchtext import transforms
from torchtext.vocab import vocab
from collections import OrderedDict

from .common.torchtext_test_case import TorchtextTestCase
from .common.assets import get_asset_path


class TestTransforms(TorchtextTestCase):
    def test_spmtokenizer_transform(self):
        asset_name = "spm_example.model"
        asset_path = get_asset_path(asset_name)
        transform = transforms.SpmTokenizerTransform(asset_path)
        actual = transform(["Hello World!, how are you?"])
        expected = [['▁Hello', '▁World', '!', ',', '▁how', '▁are', '▁you', '?']]
        self.assertEqual(actual, expected)

    def test_spmtokenizer_transform_jit(self):
        asset_name = "spm_example.model"
        asset_path = get_asset_path(asset_name)
        transform = transforms.SpmTokenizerTransform(asset_path)
        transform_jit = torch.jit.script(transform)
        actual = transform_jit(["Hello World!, how are you?"])
        expected = [['▁Hello', '▁World', '!', ',', '▁how', '▁are', '▁you', '?']]
        self.assertEqual(actual, expected)

    def test_vocab_transform(self):
        vocab_obj = vocab(OrderedDict([('a', 1), ('b', 1), ('c', 1)]))
        transform = transforms.VocabTransform(vocab_obj)
        actual = transform([['a', 'b', 'c']])
        expected = [[0, 1, 2]]
        self.assertEqual(actual, expected)
