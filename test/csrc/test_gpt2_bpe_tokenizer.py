import regex as re
import torch
import torchtext  # noqa: F401

from ..common.torchtext_test_case import TorchtextTestCase


class TestGPT2BPETokenizer(TorchtextTestCase):
    def test_gpt2_bpe_pre_tokenizer(self):
        # Regex pattern for GPT-2 BPE which includes the negative lookahead
        # Reference: https://github.com/pytorch/fairseq/blob/main/fairseq/data/encoders/gpt2_bpe_utils.py#L69
        gpt2_bpe_pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        test_cases = [
            # test spaces
            "Lorem ipsum dolor sit amet.",
            "Lorem ipsum        dolor sit amet.",
            "Lorem ipsum        dolor sit amet.  ",
            "Lorem ipsum dolor sit amet   ",
            "Lorem\x0d\x0dipsum dolor sit amet\r\r",
            "Lorem ipsum\x20dolor sit amet",
            "Lorem ipsum\x20\x20\x20dolor sit amet",
            "Lorem ipsum\x20\x20 dolor sit amet",
            # test tabs
            "Lorem ipsum dolor sit \t\t\t amet.",
            "Lorem ipsum dolor sit \t\t\t\tamet.",
            "Lorem ipsum dolor sit \x09\x09amet.",
            "Lorem ipsum dolor sit \x09\x09 amet.",
            "Lorem ipsum dolor sit \x09\x09 amet.   ",
            "Lorem ipsum dolor sit \t   \tamet.",
            "Lorem ipsum dolor sit amet   \t",
            "Lorem ipsum\tdolor sit amet",
            # test carriage returns
            "Lorem ipsum\r\r dolor sit amet",
            "Lorem ipsum\r\r dolor sit amet\r\r",
            "Lorem ipsum \x0d\x0ddolor sit amet.",
            "Lorem ipsum\x0ddolor sit amet.",
            "Lorem ipsum\x0d\x0d dolor sit amet.",
            "Lorem ipsum\x0d\x0d dolor sit amet.\x0d",
            # test form feeds
            "Lorem ipsum\f\fdolor sit amet\f",
            "Lorem ipsum\f\f dolor sit amet\f ",
            "Lorem ipsum\x0c\x0c dolor sit amet",
            "Lorem \x0c\x0c\x0c\x0cipsum dolor sit amet",
            # test vertical tabs
            "Lorem ipsum dolor sit\vamet.",
            "Lorem ipsum dolor sit\v\vamet.",
            "Lorem ipsum dolor sit\v\v amet.",
            "Lorem ipsum dolor sit\v\v amet.  \v",
            "Lorem ipsum dolor sit\x0b\x0b amet.  \v ",
            "Lorem ipsum dolor sit\x0bamet.",
            "Lorem ipsum dolor sit\x0b\x0bamet.",
            "Lorem ipsum dolor sit\x0b\x0b amet.",
        ]
        for t in test_cases:
            self.assertEqual(re.findall(gpt2_bpe_pattern, t), torch.ops.torchtext.gpt2_bpe_pre_tokenizer(t))
