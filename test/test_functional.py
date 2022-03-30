import torch
from torchtext import functional

from .common.parameterized_utils import nested_params
from .common.torchtext_test_case import TorchtextTestCase


class TestFunctional(TorchtextTestCase):
    @nested_params(
        [True, False],
        [
            [[[1, 2], [1, 2, 3]], 0, [[1, 2, 0], [1, 2, 3]]],
            [[[1, 2], [1, 2, 3]], 1, [[1, 2, 1], [1, 2, 3]]],
            [[1, 2], 0, [1, 2]],
        ],
    )
    def test_to_tensor(self, test_scripting, configs):
        """test tensorization on both single sequence and batch of sequence"""
        inputs, padding_value, expected_list = configs
        func = functional.to_tensor
        if test_scripting:
            func = torch.jit.script(func)

        actual = func(inputs, padding_value=padding_value)
        expected = torch.tensor(expected_list, dtype=torch.long)
        torch.testing.assert_close(actual, expected)

    def test_to_tensor_assert_raises(self):
        """test raise type error if input provided is not in Union[List[int],List[List[int]]]"""
        with self.assertRaises(TypeError):
            functional.to_tensor("test")

    @nested_params(
        [True, False],
        [
            [[[1, 2], [1, 2, 3]], [[1, 2], [1, 2]]],
            [[1, 2, 3], [1, 2]],
            [[["a", "b"], ["a", "b", "c"]], [["a", "b"], ["a", "b"]]],
            [["a", "b", "c"], ["a", "b"]],
        ],
    )
    def test_truncate(self, test_scripting, configs):
        """test truncation to max_seq_len length on both sequence and batch of sequence with both str/int types"""
        inputs, expected = configs
        max_seq_len = 2
        func = functional.truncate
        if test_scripting:
            func = torch.jit.script(func)

        actual = func(inputs, max_seq_len=max_seq_len)
        self.assertEqual(actual, expected)

    def test_truncate_assert_raises(self):
        """test raise type error if input provided is not in Union[List[Union[str, int]], List[List[Union[str, int]]]]"""
        with self.assertRaises(TypeError):
            functional.truncate("test", max_seq_len=2)

    @nested_params(
        [True, False],
        [
            # case: List[List[int]]
            [[[1, 2], [1, 2, 3]], 0, [[0, 1, 2], [0, 1, 2, 3]], True],
            [[[1, 2], [1, 2, 3]], 0, [[1, 2, 0], [1, 2, 3, 0]], False],
            # case: List[int]
            [[1, 2], 0, [0, 1, 2], True],
            [[1, 2], 0, [1, 2, 0], False],
            # case: List[List[str]]
            [[["a", "b"], ["c", "d"]], "x", [["x", "a", "b"], ["x", "c", "d"]], True],
            [[["a", "b"], ["c", "d"]], "x", [["a", "b", "x"], ["c", "d", "x"]], False],
            # case: List[str]
            [["a", "b"], "x", ["x", "a", "b"], True],
            [["a", "b"], "x", ["a", "b", "x"], False],
        ],
    )
    def test_add_token(self, test_scripting, configs):
        inputs, token_id, expected, begin = configs
        func = functional.add_token
        if test_scripting:
            func = torch.jit.script(func)

        actual = func(inputs, token_id=token_id, begin=begin)
        self.assertEqual(actual, expected)
