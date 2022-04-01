# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import json
import torcharrow as ta
import torcharrow.dtypes as dt
from torcharrow import functional
import torchtext.transforms as T
import torcharrow._torcharrow as _ta
from torchtext.utils import get_asset_local_path
from .common.assets import get_asset_path


# copied from GPT2BPETokenizer __init__ method
def init_bpe_encoder():
    encoder_json_path = get_asset_path("gpt2_bpe_encoder.json")
    vocab_bpe_path = get_asset_path("gpt2_bpe_vocab.bpe")
    _seperator = "\u0001"

    # load bpe encoder and bpe decoder
    with open(get_asset_local_path(encoder_json_path), "r", encoding="utf-8") as f:
        bpe_encoder = json.load(f)
    # load bpe vocab
    with open(get_asset_local_path(vocab_bpe_path), "r", encoding="utf-8") as f:
        bpe_vocab = f.read()
    bpe_merge_ranks = {
        _seperator.join(merge_pair.split()): i for i, merge_pair in enumerate(bpe_vocab.split("\n")[1:-1])
    }
    # Caching is enabled in Eager mode
    bpe = _ta.GPT2BPEEncoder(bpe_encoder, bpe_merge_ranks, _seperator, T.bytes_to_unicode(), True)
    return bpe


class _TestFunctionalBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.tokenizer = init_bpe_encoder()

        cls.base_df = ta.dataframe(
            {
                "text": ["Hello World!, how are you?", "Respublica superiorem"],
                "labels": [0, 1],
                "tokens": [["15496", "2159", "28265", "703", "389", "345", "30"], ["4965", "11377", "64", "2208", "72", "29625"]],
            },
            dtype=dt.Struct(
                fields=[
                    dt.Field("text", dt.string),
                    dt.Field("labels", dt.int32),
                    dt.Field("tokens", dt.List(dt.string)),
                ]
            ),
        )
        cls.setUpTestCaseData()

    @classmethod
    def setUpTestCaseData(cls):
        # Override in subclass
        # Python doesn't have native "abstract base test" support.
        # So use unittest.SkipTest to skip in base class: https://stackoverflow.com/a/59561905.
        raise unittest.SkipTest("abstract base test")

    def test_bpe_encode(self):
        out_df = functional.bpe_tokenize(self.tokenizer, self.df["text"])
        self.assertEqual(out_df, self.df["tokens"])


class TestFunctionalCpu(_TestFunctionalBase):
    @classmethod
    def setUpTestCaseData(cls):
        cls.df = cls.base_df.copy()


if __name__ == "__main__":
    unittest.main()
