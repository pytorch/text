# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torcharrow as ta
import torcharrow.dtypes as dt
from torcharrow import functional
# import torchtext.transforms as T
import torcharrow._torchtext


class _TestFunctionalBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        encoder_json_path = "https://download.pytorch.org/models/text/gpt2_bpe_encoder.json"
        vocab_bpe_path = "https://download.pytorch.org/models/text/gpt2_bpe_vocab.bpe"
        cls.tokenizer = functional.GPT2BPETokenizer(encoder_json_path, vocab_bpe_path)
        
        cls.base_df1 = ta.dataframe(
            {
                "text": ["hello world!", "how do you do?"],
                "labels": [0, 1]
            },
            dtype=dt.Struct(
                fields=[
                    dt.Field("text", dt.List(dt.string)),
                    dt.Field("a", dt.int32),
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
        df = type(self).df_int_float

        buckets = [2.0, 5.0, 10.0]
        self.assertEqual(
            list(functional.bucketize(df["a"], buckets)), [0, 0, 1, 1, 2, 2, 3]
        )


class TestFunctionalCpu(_TestFunctionalBase):
    @classmethod
    def setUpTestCaseData(cls):
        cls.df1 = cls.base_df2.copy()


if __name__ == "__main__":
    unittest.main()
