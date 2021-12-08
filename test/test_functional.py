import torch
from torchtext import functional
from .common.torchtext_test_case import TorchtextTestCase


class TestFunctional(TorchtextTestCase):
    def _to_tensor(self, test_scripting):
        input = [[1, 2], [1, 2, 3]]
        padding_value = 0

        func = functional.to_tensor
        if test_scripting:
            func = torch.jit.script(func)
        actual = func(input, padding_value=padding_value)
        expected = torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.long)
        torch.testing.assert_close(actual, expected)

        input = [1, 2]
        actual = func(input, padding_value=padding_value)
        expected = torch.tensor([1, 2], dtype=torch.long)
        torch.testing.assert_close(actual, expected)

    def test_to_tensor(self):
        """test tensorization on both single sequence and batch of sequence"""
        self._to_tensor(test_scripting=False)

    def test_to_tensor_jit(self):
        """test tensorization with scripting on both single sequence and batch of sequence"""
        self._to_tensor(test_scripting=True)

    def _truncate(self, test_scripting):
        max_seq_len = 2
        func = functional.truncate
        if test_scripting:
            func = torch.jit.script(func)

        input = [[1, 2], [1, 2, 3]]
        actual = func(input, max_seq_len=max_seq_len)
        expected = [[1, 2], [1, 2]]
        self.assertEqual(actual, expected)

        input = [1, 2, 3]
        actual = func(input, max_seq_len=max_seq_len)
        expected = [1, 2]
        self.assertEqual(actual, expected)

        input = [["a", "b"], ["a", "b", "c"]]
        actual = func(input, max_seq_len=max_seq_len)
        expected = [["a", "b"], ["a", "b"]]
        self.assertEqual(actual, expected)

        input = ["a", "b", "c"]
        actual = func(input, max_seq_len=max_seq_len)
        expected = ["a", "b"]
        self.assertEqual(actual, expected)

    def test_truncate(self):
        """test truncation on both sequence and batch of sequence with both str and int types"""
        self._truncate(test_scripting=False)

    def test_truncate_jit(self):
        """test truncation with scripting on both sequence and batch of sequence with both str and int types"""
        self._truncate(test_scripting=True)

    def _add_token(self, test_scripting):

        func = functional.add_token
        if test_scripting:
            func = torch.jit.script(func)
        input = [[1, 2], [1, 2, 3]]
        token_id = 0
        actual = func(input, token_id=token_id)
        expected = [[0, 1, 2], [0, 1, 2, 3]]
        self.assertEqual(actual, expected)

        actual = func(input, token_id=token_id, begin=False)
        expected = [[1, 2, 0], [1, 2, 3, 0]]
        self.assertEqual(actual, expected)

        input = [1, 2]
        actual = func(input, token_id=token_id, begin=False)
        expected = [1, 2, 0]
        self.assertEqual(actual, expected)

    def test_add_token(self):
        self._add_token(test_scripting=False)

    def test_add_token_jit(self):
        self._add_token(test_scripting=True)
