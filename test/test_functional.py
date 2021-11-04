import torch
from torchtext import functional
from .common.torchtext_test_case import TorchtextTestCase


class TestFunctional(TorchtextTestCase):
    def test_to_tensor(self):
        input = [[1, 2], [1, 2, 3]]
        padding_value = 0

        actual = functional.to_tensor(input, padding_value=padding_value)
        expected = torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.long)
        torch.testing.assert_close(actual, expected)

        input = [1, 2]
        actual = functional.to_tensor(input, padding_value=padding_value)
        expected = torch.tensor([1, 2], dtype=torch.long)
        torch.testing.assert_close(actual, expected)

    def test_to_tensor_jit(self):
        input = [[1, 2], [1, 2, 3]]
        padding_value = 0
        to_tensor_jit = torch.jit.script(functional.to_tensor)
        actual = to_tensor_jit(input, padding_value=padding_value)
        expected = torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.long)
        torch.testing.assert_close(actual, expected)

        input = [1, 2]
        actual = to_tensor_jit(input, padding_value=padding_value)
        expected = torch.tensor([1, 2], dtype=torch.long)
        torch.testing.assert_close(actual, expected)

    def test_truncate(self):
        input = [[1, 2], [1, 2, 3]]
        max_seq_len = 2

        actual = functional.truncate(input, max_seq_len=max_seq_len)
        expected = [[1, 2], [1, 2]]
        self.assertEqual(actual, expected)

        input = [1, 2, 3]
        actual = functional.truncate(input, max_seq_len=max_seq_len)
        expected = [1, 2]
        self.assertEqual(actual, expected)

    def test_truncate_jit(self):
        input = [[1, 2], [1, 2, 3]]
        max_seq_len = 2
        truncate_jit = torch.jit.script(functional.truncate)
        actual = truncate_jit(input, max_seq_len=max_seq_len)
        expected = [[1, 2], [1, 2]]
        self.assertEqual(actual, expected)

        input = [1, 2, 3]
        actual = truncate_jit(input, max_seq_len=max_seq_len)
        expected = [1, 2]
        self.assertEqual(actual, expected)

    def test_add_token(self):
        input = [[1, 2], [1, 2, 3]]
        token_id = 0
        actual = functional.add_token(input, token_id=token_id)
        expected = [[0, 1, 2], [0, 1, 2, 3]]
        self.assertEqual(actual, expected)

        actual = functional.add_token(input, token_id=token_id, begin=False)
        expected = [[1, 2, 0], [1, 2, 3, 0]]
        self.assertEqual(actual, expected)

        input = [1, 2]
        actual = functional.add_token(input, token_id=token_id, begin=False)
        expected = [1, 2, 0]
        self.assertEqual(actual, expected)

    def test_add_token_jit(self):
        input = [[1, 2], [1, 2, 3]]
        token_id = 0
        add_token_jit = torch.jit.script(functional.add_token)
        actual = add_token_jit(input, token_id=token_id)
        expected = [[0, 1, 2], [0, 1, 2, 3]]
        self.assertEqual(actual, expected)

        actual = add_token_jit(input, token_id=token_id, begin=False)
        expected = [[1, 2, 0], [1, 2, 3, 0]]
        self.assertEqual(actual, expected)

        input = [1, 2]
        actual = add_token_jit(input, token_id=token_id, begin=False)
        expected = [1, 2, 0]
        self.assertEqual(actual, expected)
