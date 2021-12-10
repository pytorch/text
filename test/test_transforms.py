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
        """test tokenization on single sentence input as well as batch on sentences"""
        self._spmtokenizer(test_scripting=False)

    def test_spmtokenizer_jit(self):
        """test tokenization with scripting on single sentence input as well as batch on sentences"""
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
        """test token to indices on both sequence of input tokens as well as batch of sequence"""
        self._vocab_transform(test_scripting=False)

    def test_vocab_transform_jit(self):
        """test token to indices with scripting on both sequence of input tokens as well as batch of sequence"""
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
        """test tensorization on both single sequence and batch of sequence"""
        self._totensor(test_scripting=False)

    def test_totensor_jit(self):
        """test tensorization with scripting on both single sequence and batch of sequence"""
        self._totensor(test_scripting=True)

    def _labeltoindex(self, test_scripting):
        label_names = ['test', 'label', 'indices']
        transform = transforms.LabelToIndex(label_names=label_names)
        if test_scripting:
            transform = torch.jit.script(transform)
        actual = transform(label_names)
        expected = [0, 1, 2]
        self.assertEqual(actual, expected)

        with self.assertRaises(RuntimeError):
            transform(['OOV'])

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
        """test labe to ids on single label input as well as batch of labels"""
        self._labeltoindex(test_scripting=False)

    def test_labeltoindex_jit(self):
        """test labe to ids with scripting on single label input as well as batch of labels"""
        self._labeltoindex(test_scripting=True)

    def _truncate(self, test_scripting):
        max_seq_len = 2
        transform = transforms.Truncate(max_seq_len=max_seq_len)
        if test_scripting:
            transform = torch.jit.script(transform)

        input = [[1, 2], [1, 2, 3]]
        actual = transform(input)
        expected = [[1, 2], [1, 2]]
        self.assertEqual(actual, expected)

        input = [1, 2, 3]
        actual = transform(input)
        expected = [1, 2]
        self.assertEqual(actual, expected)

        input = [["a", "b"], ["a", "b", "c"]]
        actual = transform(input)
        expected = [["a", "b"], ["a", "b"]]
        self.assertEqual(actual, expected)

        input = ["a", "b", "c"]
        actual = transform(input)
        expected = ["a", "b"]
        self.assertEqual(actual, expected)

    def test_truncate(self):
        """test truncation on both sequence and batch of sequence with both str and int types"""
        self._truncate(test_scripting=False)

    def test_truncate_jit(self):
        """test truncation with scripting on both sequence and batch of sequence with both str and int types"""
        self._truncate(test_scripting=True)
