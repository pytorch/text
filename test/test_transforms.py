import torch
from torchtext import transforms
from torchtext.vocab import vocab
from collections import OrderedDict

from .common.torchtext_test_case import TorchtextTestCase
from .common.assets import get_asset_path


class TestTransforms(TorchtextTestCase):
    def test_spmtokenizer(self):
        asset_name = "spm_example.model"
        asset_path = get_asset_path(asset_name)
        transform = transforms.SentencePieceTokenizer(asset_path)
        actual = transform(["Hello World!, how are you?"])
        expected = [['▁Hello', '▁World', '!', ',', '▁how', '▁are', '▁you', '?']]
        self.assertEqual(actual, expected)

    def test_spmtokenizer_jit(self):
        asset_name = "spm_example.model"
        asset_path = get_asset_path(asset_name)
        transform = transforms.SentencePieceTokenizer(asset_path)
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

    def test_totensor(self):
        input = [[1, 2], [1, 2, 3]]
        padding_value = 0
        transform = transforms.ToTensor(padding_value=padding_value)
        actual = transform(input)
        expected = torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.long)
        torch.testing.assert_close(actual, expected)

    def test_totensor_jit(self):
        input = [[1, 2], [1, 2, 3]]
        padding_value = 0
        transform = transforms.ToTensor(padding_value=padding_value)
        transform_jit = torch.jit.script(transform)
        actual = transform_jit(input)
        expected = torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.long)
        torch.testing.assert_close(actual, expected)

    def test_labeltoindex(self):
        label_names = ['test', 'label', 'indices']
        transform = transforms.LabelToIndex(label_names=label_names)
        actual = transform(label_names)
        expected = [0, 1, 2]
        self.assertEqual(actual, expected)

        transform = transforms.LabelToIndex(label_names=label_names, sort_names=True)
        actual = transform(label_names)
        expected = [2, 1, 0]
        self.assertEqual(actual, expected)

    def test_labeltoindex_jit(self):
        label_names = ['test', 'label', 'indices']
        transform_jit = torch.jit.script(transforms.LabelToIndex(label_names=label_names))
        actual = transform_jit(label_names)
        expected = [0, 1, 2]
        self.assertEqual(actual, expected)

        transform_jit = torch.jit.script(transforms.LabelToIndex(label_names=label_names, sort_names=True))
        actual = transform_jit(label_names)
        expected = [2, 1, 0]
        self.assertEqual(actual, expected)
