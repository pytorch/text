import torch
from torchtext.models import T5Transform
from torchtext_unittest.common.assets import get_asset_path
from torchtext_unittest.common.torchtext_test_case import TorchtextTestCase


class TestTransforms(TorchtextTestCase):
    def _t5tokenizer(self, test_scripting):
        asset_name = "t5_tokenizer_base.model"
        asset_path = get_asset_path(asset_name)
        transform = T5Transform(asset_path, max_seq_len=512, eos_idx=1, padding_idx=0)
        if test_scripting:
            transform = torch.jit.script(transform)

        # test encode; input is a single string
        encode_seq = "Hello World!, how are you?"
        actual = transform(encode_seq)
        expected = torch.tensor([8774, 1150, 55, 6, 149, 33, 25, 58, 1])
        self.assertEqual(actual, expected)

        # test encode; input is a batched string
        encode_seq = ["Hello World!, how are you?"]
        actual = transform(encode_seq)
        expected = torch.tensor([[8774, 1150, 55, 6, 149, 33, 25, 58, 1]])
        self.assertEqual(actual, expected)

        # test decode; input is a list of token ids
        decode_seq = [8774, 1150, 55, 6, 149, 33, 25, 58, 1]
        actual = transform.decode(decode_seq)
        expected = "Hello World!, how are you?"
        self.assertEqual(actual, expected)

        # test decode; input is a batched list of token ids
        decode_seq = [[8774, 1150, 55, 6, 149, 33, 25, 58, 1]]
        actual = transform.decode(decode_seq)
        expected = ["Hello World!, how are you?"]
        self.assertEqual(actual, expected)

    def test_t5tokenizer(self) -> None:
        """test tokenization on string input (encode) and translation from token ids to strings (decode)"""
        self._t5tokenizer(test_scripting=False)

    def test_t5tokenizer_jit(self) -> None:
        """test tokenization on string input (encode) and translation from token ids to strings (decode) with scripting"""
        self._t5tokenizer(test_scripting=True)
