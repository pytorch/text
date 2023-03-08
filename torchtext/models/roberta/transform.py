from typing import List, Union
from urllib.parse import urljoin

import torch
import torchtext.transforms as T
from torch.nn import Module
from torchtext import _TEXT_BUCKET
from torchtext._download_hooks import load_state_dict_from_url


class RobertaTransform(Module):
    """Standard transform for RoBERTa model."""

    def __init__(self, truncate_length: int) -> None:
        super().__init__()
        self.transform = T.Sequential(
            T.GPT2BPETokenizer(
                encoder_json_path=urljoin(_TEXT_BUCKET, "gpt2_bpe_encoder.json"),
                vocab_bpe_path=urljoin(_TEXT_BUCKET, "gpt2_bpe_vocab.bpe"),
            ),
            T.VocabTransform(load_state_dict_from_url(urljoin(_TEXT_BUCKET, "roberta.vocab.pt"))),
            T.Truncate(truncate_length),
            T.AddToken(token=0, begin=True),
            T.AddToken(token=2, begin=False),
            T.ToTensor(padding_value=0),
        )

    def forward(self, inputs: Union[str, List[str]]) -> torch.Tensor:
        transformed_outputs = self.transform(inputs)
        assert torch.jit.isinstance(transformed_outputs, torch.Tensor)
        return transformed_outputs


class XLMRTransform(Module):
    """Standard transform for XLMR model."""

    def __init__(self, truncate_length: int) -> None:
        super().__init__()
        self.tokenizer = T.SentencePieceTokenizer(urljoin(_TEXT_BUCKET, "xlmr.sentencepiece.bpe.model"))
        self.transform = T.Sequential(
            T.SentencePieceTokenizer(urljoin(_TEXT_BUCKET, "xlmr.sentencepiece.bpe.model")),
            T.VocabTransform(load_state_dict_from_url(urljoin(_TEXT_BUCKET, "xlmr.vocab.pt"))),
            T.Truncate(truncate_length),
            T.AddToken(token=0, begin=True),
            T.AddToken(token=2, begin=False),
            T.ToTensor(padding_value=0),
        )

    def forward(self, inputs: Union[str, List[str]]) -> torch.Tensor:
        transformed_outputs = self.transform(inputs)
        assert torch.jit.isinstance(transformed_outputs, torch.Tensor)
        return transformed_outputs
