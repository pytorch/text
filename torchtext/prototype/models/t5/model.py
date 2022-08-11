from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Callable

import torch
import torch.nn as nn
from torch import Tensor

from .modules import T5Encoder, T5Decoder, T5LayerNorm


@dataclass
class T5Conf:
    encoder_only: bool = False
    linear_head: bool = False
    embedding_dim: int = 768
    qkv_dim: int = 64
    num_attention_heads: int = 12
    num_encoder_layers: int = 12
    num_decoder_layers: int = 12
    ffn_dimension: int = 3072
    dropout: float = 0.1
    activation: Union[str, Callable[[Tensor], Tensor]] = "relu"
    layer_norm_eps: float = 1e-6
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    padding_idx: int = 0
    max_seq_len: int = 512
    vocab_size: int = 32128
    training: bool = False


# NOTE: Comparable HuggingFace implentation can be found at https://github.com/huggingface/transformers/blob/8581a798c0a48fca07b29ce2ca2ef55adcae8c7e/src/transformers/models/t5/modeling_t5.py#L1269
class T5Model(nn.Module):
    r"""A T5 model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer".
    Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena,
    Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Journal of Machine Learning Research.
    Volume 21 Issue 140 pages 1-67. http://jmlr.org/papers/v21/20-074.html
    Args:
        config.encoder_only: Whether or not model should consist of only the encoder as opposed to encoder-decoder (default=False).
        config.linear_head: Whether or not a linear layer should be used to project the output of the decoder's last layer to the vocab (default=False).
        config.embedding_dim: Number of expected features in the encoder/decoder inputs (default=768).
        config.qkv_dim: Projection dimension (per head) for query, keys, and values. (defualt=64).
        config.num_attention_heads: Number of heads in the multiheadattention models (default=12).
        config.num_encoder_layers: Number of encoder layers in the encoder (default=12).
        config.num_decoder_layers: Number of decoder layers in the decoder (default=12).
        config.ffn_dimension: Dimension of the feedforward network model (default=3072).
        config.dropout: Dropout value (default=0.1).
        config.activation: Activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        config.layer_norm_eps: The eps value in layer normalization components (default=1e-6).
        config.relative_attention_num_buckets: Number of relative position buckets (default: 32)
        config.relative_attention_max_distance: Maximum threshold on the relative distance used to
            allocate buckets. Anything larger gets placed in the same bucket (default: 128)
        config.padding_idx: Index assigned to padding token in vocabulary (default: 0)
        config.max_seq_len: Maximum sequence length (default: 512)
        config.vocab_size: Size of vocabulary (default: 32128)
        config.training: Whether or not to apply dropout (default: False)
        freeze: Indicates whether or not to freeze the model weights. (default: False)
    Examples:
        >>> from torchtext.prototype.models import T5Conf, T5Model
        >>> t5_config = T5Conf(encoder_only=False, linear_head=True)
        >>> t5_model = T5Model(t5_config)
        >>> encoder_input = torch.randint(0, t5_config.vocab_size, (32, 512))
        >>> out = t5_model(encoder_input)['decoder_output']
        >>> out.shape
        torch.Size([32, 1, 32128])
    """

    def __init__(
        self,
        config: T5Conf,
        freeze: bool = False,
        device: Optional[torch.device] = None,
        dtype=None,
    ) -> None:
        super().__init__()

        assert isinstance(config, T5Conf)

        self.config = config
        self.embedding_dim = config.embedding_dim
        self.encoder_only = config.encoder_only
        self.linear_head = config.linear_head
        self.padding_idx = config.padding_idx
        self.training = config.training
        self.dropout = config.dropout if config.training else 0.0
        self.device = device
        self.dtype = dtype

        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim, config.padding_idx)
        self.encoder = T5Encoder(
            d_model=config.embedding_dim,
            nhead=config.num_attention_heads,
            num_layers=config.num_encoder_layers,
            dim_feedforward=config.ffn_dimension,
            qkv_dim=config.qkv_dim,
            dropout=self.dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            relative_attention_num_buckets=config.relative_attention_num_buckets,
            relative_attention_max_distance=config.relative_attention_max_distance,
            device=device,
            dtype=dtype,
        )
        self.norm1 = T5LayerNorm(config.embedding_dim)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

        if not config.encoder_only:
            self.decoder = T5Decoder(
                d_model=config.embedding_dim,
                nhead=config.num_attention_heads,
                num_layers=config.num_decoder_layers,
                dim_feedforward=config.ffn_dimension,
                qkv_dim=config.qkv_dim,
                dropout=self.dropout,
                activation=config.activation,
                layer_norm_eps=config.layer_norm_eps,
                relative_attention_num_buckets=config.relative_attention_num_buckets,
                relative_attention_max_distance=config.relative_attention_max_distance,
                device=device,
                dtype=dtype,
            )
            self.norm2 = T5LayerNorm(config.embedding_dim)
            self.dropout3 = nn.Dropout(self.dropout)
            self.dropout4 = nn.Dropout(self.dropout)
        else:
            self.decoder = None

        if config.linear_head:
            self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(
        self,
        encoder_tokens: Tensor,
        decoder_tokens: Optional[Tensor] = None,
        encoder_mask: Optional[Tensor] = None,
        decoder_mask: Optional[Tensor] = None,
    ) -> Dict[str, Union[Tensor, List[Tensor], Optional[Tensor], List[Optional[Tensor]]]]:
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        Args:
            encoder_tokens: Tokenized input sequence to the encoder.
                Must be batch first with shape (B, Ne) where B is the batch size and Ne is the
                encoder input sequence length. (required).
            decoder_tokens: Tokenized input sequence to the decoder.
                Must be batch first with shape (B, Nd) where B is the batch size and Nd is the
                decoder input sequence length. If None and model is encoder-decoder, will initialize decoder
                input sequence to begin with padding index. (optional).
            encoder_mask: Self-attention mask for the encoder input sequence.
                Must have shape (Ne, Ne) (optional).
            decoder_mask: Self-attention mask for the decoder input sequence.
                Must have shape (Nd, Nd) (optional).
        Returns:
            encoder_output: Output Tensor from the final layer of the encoder
            encoder_hidden_states: Tuple of output Tensors from each layer of the encoder
            encoder_position_bias: Tensor of relative attention bias computed for input sequence to encoder
            encoder_sa_scores: Tuple of self-attention scores computed at each layer of the encoder
            decoder_output: Output Tensor from the final layer of the decoder
            decoder_hidden_states: Tuple of output Tensors from each layer of the decoder
            decoder_position_bias: Tensor of relative attention bias computed for input sequence to decoder
            encoder_sa_scores: Tuple of self-attention scores computed at each layer of the decoder
            encoder_ca_scores: Tuple of cross-attention scores computed at each layer of the decoder
        """
        encoder_padding_mask = encoder_tokens.eq(self.padding_idx)
        encoder_embeddings = self.dropout1(self.token_embeddings(encoder_tokens))
        encoder_output, encoder_hidden_states, encoder_position_bias, encoder_sa = self.encoder(
            encoder_embeddings, tgt_mask=encoder_mask, tgt_key_padding_mask=encoder_padding_mask
        )

        encoder_output = self.norm1(encoder_output)
        encoder_output = self.dropout2(encoder_output)
        encoder_hidden_states.append(encoder_output)

        if not self.encoder_only:

            assert self.decoder is not None

            # decoder_tokens is None means at start of inference, in which case decoder sequence should begin with padding idx.
            if decoder_tokens is None:
                decoder_tokens = torch.ones((encoder_tokens.size(0), 1), dtype=torch.long) * self.padding_idx

            if decoder_mask is None:
                assert decoder_tokens is not None and decoder_tokens.dim() == 2
                tgt_len = decoder_tokens.shape[1]
                decoder_mask = torch.triu(torch.ones((tgt_len, tgt_len), dtype=torch.float64), diagonal=1)
                decoder_mask = decoder_mask.to(torch.bool)

            decoder_padding_mask = decoder_tokens.eq(self.padding_idx)
            # T5 implemention uses padding idx to start sequence. Want to ignore this when masking
            decoder_padding_mask[:, 0] = False

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
            decoder_hidden_states.append(decoder_output)

            if self.linear_head:
                assert self.lm_head is not None
                # Rescale output before projecting on vocab. This happens when the encoder and decoder share the
                # same word embeddings, which is always the case in our t5 implementation.
                # See https://github.com/huggingface/transformers/blob/d0acc9537829e7d067edbb791473bbceb2ecf056/src/transformers/models/t5/modeling_t5.py#L1661
                decoder_output = decoder_output * (self.embedding_dim ** -0.5)
                decoder_output = self.lm_head(decoder_output)

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
        else:
            t5_output = {
                "encoder_output": encoder_output,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_position_bias": encoder_position_bias,
                "encoder_sa_scores": encoder_sa,
            }

            assert torch.jit.isinstance(
                t5_output, Dict[str, Union[Tensor, List[Tensor], Optional[Tensor], List[Optional[Tensor]]]]
            )

        return t5_output
