import torch
from torchtext import transforms
from torchtext.vocab import vocab
from collections import OrderedDict

from .common.torchtext_test_case import TorchtextTestCase
from .common.assets import get_asset_path


class TestTransforms(TorchtextTestCase):
    def _spmtokenizer(self, test_scripting):
        asset_name = "spm_example.model"
        asset_path = get_asset_path(asset_name)
        transform = transforms.SentencePieceTokenizer(asset_path)
        if test_scripting:
            transform = torch.jit.script(transform)
        actual = transform(["Hello World!, how are you?"])
        expected = [['▁Hello', '▁World', '!', ',', '▁how', '▁are', '▁you', '?']]
        self.assertEqual(actual, expected)

        actual = transform("Hello World!, how are you?")
        expected = ['▁Hello', '▁World', '!', ',', '▁how', '▁are', '▁you', '?']
        self.assertEqual(actual, expected)

    def test_spmtokenizer(self):

        self._spmtokenizer(test_scripting=False)

    def test_spmtokenizer_jit(self):
        self._spmtokenizer(test_scripting=True)

    def _vocab_transform(self, test_scripting):
        vocab_obj = vocab(OrderedDict([('a', 1), ('b', 1), ('c', 1)]))
        transform = transforms.VocabTransform(vocab_obj)
        if test_scripting:
            transform = torch.jit.script(transform)
        actual = transform([['a', 'b', 'c']])
        expected = [[0, 1, 2]]
        self.assertEqual(actual, expected)

        actual = transform(['a', 'b', 'c'])
        expected = [0, 1, 2]
        self.assertEqual(actual, expected)

    def test_vocab_transform(self):
        self._vocab_transform(test_scripting=False)

    def test_vocab_transform_jit(self):
        self._vocab_transform(test_scripting=True)

    def _totensor(self, test_scripting):
        padding_value = 0
        transform = transforms.ToTensor(padding_value=padding_value)
        if test_scripting:
            transform = torch.jit.script(transform)
        input = [[1, 2], [1, 2, 3]]

        actual = transform(input)
        expected = torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.long)
        torch.testing.assert_close(actual, expected)

        input = [1, 2]
        actual = transform(input)
        expected = torch.tensor([1, 2], dtype=torch.long)
        torch.testing.assert_close(actual, expected)

    def test_totensor(self):
        self._totensor(test_scripting=False)

    def test_totensor_jit(self):
        self._totensor(test_scripting=True)

    def _labeltoindex(self, test_scripting):
        label_names = ['test', 'label', 'indices']
        transform = transforms.LabelToIndex(label_names=label_names)
        if test_scripting:
            transform = torch.jit.script(transform)
        actual = transform(label_names)
        expected = [0, 1, 2]
        self.assertEqual(actual, expected)

        transform = transforms.LabelToIndex(label_names=label_names, sort_names=True)
        if test_scripting:
            transform = torch.jit.script(transform)
        actual = transform(label_names)
        expected = [2, 1, 0]
        self.assertEqual(actual, expected)

        actual = transform("indices")
        expected = 0
        self.assertEqual(actual, expected)

        asset_name = "label_names.txt"
        asset_path = get_asset_path(asset_name)
        transform = transforms.LabelToIndex(label_path=asset_path)
        if test_scripting:
            transform = torch.jit.script(transform)
        actual = transform(label_names)
        expected = [0, 1, 2]
        self.assertEqual(actual, expected)

    def test_labeltoindex(self):
        self._labeltoindex(test_scripting=False)

    def test_labeltoindex_jit(self):
        self._labeltoindex(test_scripting=True)
