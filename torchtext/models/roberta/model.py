import logging
import math
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from .modules import TransformerEncoder

logger = logging.getLogger(__name__)


@dataclass
class RobertaEncoderConf:
    vocab_size: int = 50265
    embedding_dim: int = 768
    ffn_dimension: int = 3072
    padding_idx: int = 1
    max_seq_len: int = 514
    num_attention_heads: int = 12
    num_encoder_layers: int = 12
    dropout: float = 0.1
    scaling: Optional[float] = None
    normalize_before: bool = False


class RobertaEncoder(Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        ffn_dimension: int,
        padding_idx: int,
        max_seq_len: int,
        num_attention_heads: int,
        num_encoder_layers: int,
        dropout: float = 0.1,
        scaling: Optional[float] = None,
        normalize_before: bool = False,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        if not scaling:
            head_dim = embedding_dim // num_attention_heads
            scaling = 1.0 / math.sqrt(head_dim)

        self.transformer = TransformerEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_seq_len=max_seq_len,
            ffn_dimension=ffn_dimension,
            num_encoder_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            normalize_before=normalize_before,
            scaling=scaling,
            return_all_layers=False,
        )

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, tokens: Tensor, masked_tokens: Optional[Tensor] = None) -> Tensor:
        output = self.transformer(tokens)
        if torch.jit.isinstance(output, List[Tensor]):
            output = output[-1]
        output = output.transpose(1, 0)
        if masked_tokens is not None:
            output = output[masked_tokens.to(torch.bool), :]
        return output


# TODO: Add Missing quant noise and spectral norm from latest Roberta head in fairseq repo
class RobertaClassificationHead(nn.Module):
    def __init__(
        self, num_classes, input_dim, inner_dim: Optional[int] = None, dropout: float = 0.1, activation=nn.ReLU
    ) -> None:
        super().__init__()
        if not inner_dim:
            inner_dim = input_dim
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
        self.activation_fn = activation()

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaModel(Module):
    """

    Example - Instatiating model object
        >>> from torchtext.models import RobertaEncoderConf, RobertaModel, RobertaClassificationHead
        >>> roberta_encoder_conf = RobertaEncoderConf(vocab_size=250002)
        >>> encoder = RobertaModel(config=roberta_encoder_conf)
        >>> classifier_head = RobertaClassificationHead(num_classes=2, input_dim=768)
        >>> classifier = RobertaModel(config=roberta_encoder_conf, head=classifier_head)
    """

    def __init__(
        self, encoder_conf: RobertaEncoderConf, head: Optional[Module] = None, freeze_encoder: bool = False
    ) -> None:
        super().__init__()
        assert isinstance(encoder_conf, RobertaEncoderConf)

        self.encoder = RobertaEncoder(**asdict(encoder_conf), freeze=freeze_encoder)
        self.head = head

    def forward(self, tokens: Tensor, masked_tokens: Optional[Tensor] = None) -> Tensor:
        features = self.encoder(tokens, masked_tokens)
        if self.head is None:
            return features

        x = self.head(features)
        return x
