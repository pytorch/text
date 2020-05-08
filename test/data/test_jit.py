import torch
from torchtext.experimental.transforms import TokenizerTransform
from ..common.torchtext_test_case import TorchtextTestCase


class TestJITTransforms(TorchtextTestCase):
    def test_jit_TokenizerTransform(self):
        tok_transform = TokenizerTransform("basic_english")
        ts_tok_transform = torch.jit.script(tok_transform)
        self.assertEqual(tok_transform("here   (we)!?:; \' \"   are"),
                         ts_tok_transform("here   (we)!?:; \' \"   are"))

        tok_transform = TokenizerTransform(None)
        ts_tok_transform = torch.jit.script(tok_transform)
        self.assertEqual(tok_transform("here   (we)!?:; \' \"   are"),
                         ts_tok_transform("here   (we)!?:; \' \"   are"))
