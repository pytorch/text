import torch
from test.common.assets import get_asset_path
from test.common.torchtext_test_case import TorchtextTestCase
from torchtext.prototype.models import T5Transform


class TestTransforms(TorchtextTestCase):
    def _t5tokenizer(self, test_scripting):
        asset_name = "t5_tokenizer_base.model"
        asset_path = get_asset_path(asset_name)
        transform = T5Transform(asset_path, max_seq_len=512, eos_idx=1, padding_idx=0)
        if test_scripting:
            transform = torch.jit.script(transform)

        actual = transform(["Hello World!, how are you?"])
        expected = torch.tensor([[8774, 1150, 55, 6, 149, 33, 25, 58, 1]])
        self.assertEqual(actual, expected)

        actual = transform("Hello World!, how are you?")
        expected = torch.tensor([8774, 1150, 55, 6, 149, 33, 25, 58, 1])
        self.assertEqual(actual, expected)

    def test_t5tokenizer(self):
        """test tokenization on single sentence input as well as batch on sentences"""
        self._t5tokenizer(test_scripting=False)

    def test_t5tokenizer_jit(self):
        """test tokenization with scripting on single sentence input as well as batch on sentences"""
        self._t5tokenizer(test_scripting=True)
