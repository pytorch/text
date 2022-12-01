# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch

from torchtext.transforms import (
    AddToken,
    BERTTokenizer,
    Sequential,
    StrToIntTransform,
    ToTensor,
)


class BertTextTransform:
    """Transform raw text into a Tensor of token ids for using BERT models.

    Args:
        vocab_file (str): local or URL path to pre-trained vocab file.
            Defaults to HuggingFace BERT base (uncased) model's vocab file.
        do_lower_case (bool): whether to convert input text to lowercase.
            Defaults to True.
        start_token (int): value to represent the start of each text.
            Defaults to 101, Hugging Face's BERT start token.
        end_token (int): value to represent the end of each text.
            Defaults to 102, Hugging Face's BERT end token.
        padding_value (int): value with which to pad each text so that all texts are the same length.
            Defaults to 0, Hugging Face's BERT pad token.

    Inputs:
        raw_text (List[str]): list of raw texts to transform

    Returns:
        Tensor: Token IDs representing the input text, of dimensions
            (len(raw_text), max length of input text)
    """

    def __init__(
        self,
        vocab_file: str = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
        do_lower_case: bool = True,
        start_token: int = 101,
        end_token: int = 102,
        padding_value: int = 0,
    ):
        self.tokenizer = Sequential(
            BERTTokenizer(
                vocab_path=vocab_file, do_lower_case=do_lower_case, return_tokens=False
            ),
            StrToIntTransform(),
            AddToken(start_token, begin=True),
            AddToken(end_token, begin=False),
            ToTensor(padding_value=padding_value),
        )

    def __call__(self, raw_text: List[str]) -> torch.Tensor:
        input_ids = self.tokenizer(raw_text)
        return input_ids
