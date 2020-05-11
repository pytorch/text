import torch
from torchtext.experimental.transforms import SplitTokenizer, BasicEnglishTokenizer
from ..common.torchtext_test_case import TorchtextTestCase


class TestJITTransforms(TorchtextTestCase):
    def test_jit_SplitTokenizer(self):
        tok_transform = SplitTokenizer()
        ts_tok_transform = torch.jit.script(tok_transform)
        self.assertEqual(tok_transform("here   (we)!?:; \' \"   are"),
                         ts_tok_transform("here   (we)!?:; \' \"   are"))

    def test_jit_BasicEnglishTokenizer(self):
        tok_transform = BasicEnglishTokenizer()
        ts_tok_transform = torch.jit.script(tok_transform)
        self.assertEqual(tok_transform("here   (we)!?:; \' \"   are"),
                         ts_tok_transform("here   (we)!?:; \' \"   are"))
