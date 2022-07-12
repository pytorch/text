from typing import Optional, Union, Callable

import torch
import torch.nn as nn
from torch import Tensor

from .modules import T5Stack, T5LayerNorm


class T5Model(nn.Module):
    r"""A T5 model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer".
    Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena,
    Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Journal of Machine Learning Research.
    Volume 21 Issue 140 pages 1-67. http://jmlr.org/papers/v21/20-074.html
    Args:
        encoder_only: whether or not model should consist of only the encoder as opposed to encoder-decoder (required)
        d_model: the number of expected features in the encoder/decoder inputs (default=768.
        nhead: the number of heads in the multiheadattention models (default=12).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=12).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=12).
        dim_feedforward: the dimension of the feedforward network model (default=3072).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-6).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``True`` (seq, batch, feature).
        relative_attention_num_buckets: the number of relative position buckets (default: 32)
        relative_attention_max_distance: maximum threshold on the relative distance used to
            allocate buckets. anything larger than that gets placed in the same bucket (default: 128)
        padding_idx: index assigned to padding token in vocabulary (default: 0)
        max_seq_len: maximum sequence length (default: 512)
        vocab_size: size of vocabulary (default: 32128)
    Examples::
        >>> t5_model = T5Model(encoder_only=False)
        >>> src = torch.rand((32, 10, 512))
        >>> tgt = torch.rand((32, 20, 512))
        >>> out = t5_model(src, tgt)
    """

    def __init__(
        self,
        encoder_only: bool,
        d_model: int = 768,
        nhead: int = 12,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = "relu",
        layer_norm_eps: float = 1e-6,
        batch_first: bool = True,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        padding_idx: int = 0,
        max_seq_len: int = 512,
        vocab_size: int = 32128,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.encoder_only = encoder_only
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.batch_first = batch_first
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.realtive_attention_max_distance = relative_attention_max_distance
        self.padding_idx = padding_idx
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.device = device
        self.dtype = dtype

        self.token_embeddings = nn.Embedding(vocab_size, d_model, padding_idx)
        self.encoder = T5Stack(
            is_decoder=False,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            device=device,
            dtype=dtype,
        )
        self.norm1 = T5LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if not encoder_only:
            self.decoder = T5Stack(
                is_decoder=True,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                relative_attention_num_buckets=relative_attention_num_buckets,
                relative_attention_max_distance=relative_attention_max_distance,
                device=device,
                dtype=dtype,
            )
            self.norm2 = T5LayerNorm(d_model)
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

    def forward(
        self,
        encoder_tokens: Tensor,
        decoder_tokens: Tensor = None,
        encoder_mask: Optional[Tensor] = None,
        decoder_mask: Optional[Tensor] = None,
    ) -> Tensor:
        encoder_padding_mask = encoder_tokens.eq(self.padding_idx)
        encoder_embeddings = self.dropout1(self.token_embeddings(encoder_tokens))
        encoder_output, encoder_hidden_states, encoder_position_bias, encoder_sa, _ = self.encoder(
            encoder_embeddings, tgt_mask=encoder_mask, tgt_key_padding_mask=encoder_padding_mask
        )

        encoder_output = self.norm1(encoder_output)
        encoder_output = self.dropout2(encoder_output)
        encoder_hidden_states = encoder_hidden_states + (encoder_output,)

        decoder_output = None
        decoder_hidden_states = None
        decoder_position_bias = None

        if not self.encoder_only:
            assert decoder_tokens is not None
            if decoder_mask is None:
                tgt_len = decoder_tokens.shape[1]
                decoder_mask = torch.triu(torch.ones((tgt_len, tgt_len), dtype=torch.float64), diagonal=1).bool()

            decoder_padding_mask = decoder_tokens.eq(self.padding_idx)
            decoder_embeddings = self.dropout3(self.token_embeddings(decoder_tokens))
            decoder_output, decoder_hidden_states, decoder_position_bias, decoder_sa, decoder_ca = self.decoder(
                decoder_embeddings,
                memory=encoder_output,
                tgt_mask=decoder_mask,
                memory_mask=encoder_mask,
                tgt_key_padding_mask=decoder_padding_mask,
                memory_key_padding_mask=encoder_padding_mask,
            )

            decoder_output = self.norm2(decoder_output)
            decoder_output = self.dropout4(decoder_output)
            decoder_hidden_states = decoder_hidden_states + (decoder_output,)

        t5_output = {
            "encoder_output": encoder_output,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_position_bias": encoder_position_bias,
            "encoder_sa_scores": encoder_sa,
            "decoder_output": decoder_output,
            "decoder_hidden_states": decoder_hidden_states,
            "decoder_position_bias": decoder_position_bias,
            "decoder_sa_scores": decoder_sa,
            "decoder_ca_scores": decoder_ca,
        }

        return t5_output
