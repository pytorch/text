import math

from dataclasses import dataclass, asdict
from typing import Optional

from torch.nn import Module
import torch
from torch import Tensor
import torch.nn as nn

from .modules import (
    TransformerEncoder,
)
import logging
logger = logging.getLogger(__name__)


@dataclass
class RobertaEncoderParams:
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
    ):
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
        )

    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        all_layers = self.transformer(tokens)
        last_layer = all_layers[-1].transpose(1, 0)
        if mask is not None:
            last_layer = last_layer[mask.to(torch.bool), :]
        return last_layer


# TODO: Add Missing quant noise and spectral norm from latest Roberta head in fairseq repo
class RobertaClassificationHead(nn.Module):
    def __init__(self, num_classes, input_dim, inner_dim: Optional[int] = None, dropout: float = 0.1, activation=nn.ReLU):
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
    def __init__(self, encoder: Module, head: Optional[Module] = None):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, tokens: Tensor) -> Tensor:
        features = self.encoder(tokens)
        if self.head is None:
            return features

        x = self.head(features)
        return x


def _get_model(params: RobertaEncoderParams, head: Optional[Module] = None, freeze_encoder: bool = False) -> RobertaModel:
    encoder = RobertaEncoder(**asdict(params))
    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False

        logger.info("Encoder weights are frozen")

    return RobertaModel(encoder, head)
