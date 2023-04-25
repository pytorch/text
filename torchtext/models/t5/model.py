# /* Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. */
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .modules import SEQ_2_SEQ_OUTPUTS_TYPE, PAST_KEY_VALUES_TYPE, T5Decoder, T5Encoder


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
    feed_forward_proj: str = None
    is_gated_act: bool = False

    def __post_init__(self):
        """The following is modified from:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/configuration_t5.py

        Supports T5 1.1 and FLAN-T5.
        """
        if self.feed_forward_proj:
            act_info = self.feed_forward_proj.split("-")
            self.activation = act_info[-1]
            self.is_gated_act = act_info[0] == "gated"

            if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
                raise ValueError(
                    f"`feed_forward_proj`: {self.feed_forward_proj} is not a valid activation function of the dense layer."
                    "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                    "'gated-gelu' or 'relu'"
                )

            # for backwards compatibility
            if self.feed_forward_proj == "gated-gelu":
                self.activation = "gelu_new"


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
        >>> from torchtext.models import T5Conf, T5Model
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
            token_embeddings=self.token_embeddings,
            is_gated_act=config.is_gated_act,
            device=device,
            dtype=dtype,
        )

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
                is_gated_act=config.is_gated_act,
                device=device,
                dtype=dtype,
            )
        else:
            self.decoder = None

        if config.linear_head:
            self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    @torch.jit.export
    def _reorder_cache(self, past: List[PAST_KEY_VALUES_TYPE], beam_idx: Tensor) -> List[PAST_KEY_VALUES_TYPE]:
        """Reorder past key value pairs in cache. Only relevant in incremental decoding with beam search generation."""
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            return past

        reordered_decoder_past: List[PAST_KEY_VALUES_TYPE] = []
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past.append(reordered_layer_past_states)
        return reordered_decoder_past

    @torch.jit.export
    def _shift_right(self, input_ids: Tensor) -> Tensor:
        """Shift all input sequences to the right"""
        shifted_input_ids = torch.zeros_like(input_ids)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()

        # T5 implemention uses padding idx to start sequence.
        shifted_input_ids[:, 0] = self.padding_idx

        return shifted_input_ids

    @torch.jit.export
    def prepare_inputs_for_generation(
        self,
        input_ids: Tensor,
        encoder_outputs: Optional[SEQ_2_SEQ_OUTPUTS_TYPE] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        past: Optional[List[PAST_KEY_VALUES_TYPE]] = None,
        return_past_key_values: bool = True,
        model_kwargs: Optional[
            Dict[str, Union[SEQ_2_SEQ_OUTPUTS_TYPE, Optional[Tensor], Optional[List[PAST_KEY_VALUES_TYPE]], bool]]
        ] = None,
    ) -> Dict[str, Union[Tensor, SEQ_2_SEQ_OUTPUTS_TYPE, Optional[List[PAST_KEY_VALUES_TYPE]], bool]]:
        """Prepare inputs for generation from model_kwargs.

        Args:
            input_ids (Tensor): Seed tokens for generation.
            model_kwargs (Dict): Other specifications for generation

        Returns:
            Dictionary bundling all model inputs for generation.
        """
        if model_kwargs is None:
            assert encoder_outputs is not None
            model_kwargs = {
                "encoder_outputs": encoder_outputs,
                "encoder_padding_mask": encoder_padding_mask,
                "past_key_values": past,
                "return_past_key_values": return_past_key_values,
            }

        # Incremental decoding if past key values are provided
        past = model_kwargs.get("past", None)
        if past is not None:
            input_ids = input_ids[:, -1:]

        model_kwargs["decoder_tokens"] = input_ids
        return model_kwargs

    def get_encoder(self) -> T5Encoder:
        return self.encoder

    def get_decoder(self) -> Optional[T5Decoder]:
        if self.decoder is None:
            warnings.warn("Decoder is not set on this model.")
        return self.decoder

    def forward(
        self,
        encoder_tokens: Optional[Tensor] = None,
        decoder_tokens: Optional[Tensor] = None,
        encoder_mask: Optional[Tensor] = None,
        decoder_mask: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        decoder_padding_mask: Optional[Tensor] = None,
        encoder_outputs: Optional[SEQ_2_SEQ_OUTPUTS_TYPE] = None,
        past_key_values: Optional[List[PAST_KEY_VALUES_TYPE]] = None,
        return_past_key_values: bool = False,
    ) -> SEQ_2_SEQ_OUTPUTS_TYPE:
        r"""Pass the inputs (and mask) through the T5Encoder/T5Decoder in turn.

        Args:
            encoder_tokens: Tokenized input sequence to the encoder.
                Must be batch first with shape (B, Ne) where B is the batch size and Ne is the
                encoder input sequence length. (optional if `encoder_outputs` is provided)
            decoder_tokens: Tokenized input sequence to the decoder.
                Must be batch first with shape (B, Nd) where B is the batch size and Nd is the
                decoder input sequence length. If None and model is encoder-decoder, will initialize decoder
                input sequence to begin with padding index. (optional).
            encoder_mask: Self-attention mask for the encoder input sequence.
                Must have shape (Ne, Ne) (optional).
            decoder_mask: Self-attention mask for the decoder input sequence.
                Must have shape (Nd, Nd) (optional).
            encoder_padding_mask: Padding mask for encoder input sequence.
                Must have shape (B, Ne) (optional).
            decoder_padding_mask: Padding mask for decoder input sequence.
                Must have shape (B, Nd) (optional).
            encoder_outputs: Outputs from previous run of T5Encoder. (optional)
            past_key_values: Previously calculated key values, used in incremental decoding. (optional)
            return_past_key_values: Boolean indicating whether to return key values to user. (default: False)

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
            past_key_values: List of Tuples of key values calculated during this run, or None.
        """
        seq2seq_model_output: SEQ_2_SEQ_OUTPUTS_TYPE = {
            "encoder_output": None,
            "encoder_hidden_states": None,
            "encoder_sa_scores": None,
            "encoder_ca_scores": None,
            "decoder_output": None,
            "decoder_hidden_states": None,
            "decoder_sa_scores": None,
            "decoder_ca_scores": None,
            "past_key_values": None,
        }

        if encoder_outputs is None:
            assert encoder_tokens is not None, "If `encoder_outputs` is not specified, must provide `encoder_tokens`"

            if encoder_padding_mask is None:
                encoder_padding_mask = encoder_tokens.eq(self.padding_idx).to(device=encoder_tokens.device)

            encoder_outputs = self.encoder(
                src=encoder_tokens, mask=encoder_mask, src_key_padding_mask=encoder_padding_mask
            )

            seq2seq_model_output.update(encoder_outputs)

        if not self.encoder_only:
            assert self.decoder is not None
            assert encoder_outputs is not None

            encoder_output = encoder_outputs.get("encoder_output")
            assert torch.jit.isinstance(encoder_output, Tensor)

            batch_size = encoder_output.size(0)
            encoder_output_device = encoder_output.device

            # decoder_tokens is None means at start of inference, in which case decoder sequence should begin with padding idx.
            if decoder_tokens is None:
                decoder_tokens = (
                    torch.ones((batch_size, 1), device=encoder_output_device, dtype=torch.long) * self.padding_idx
                )

            if decoder_padding_mask is None:
                decoder_padding_mask = decoder_tokens.eq(self.padding_idx)
                # T5 implemention uses padding idx to start sequence. Want to ignore this when masking
                decoder_padding_mask[:, 0] = False

            decoder_embeddings = self.token_embeddings(decoder_tokens)
            decoder_outputs = self.decoder(
                decoder_embeddings,
                memory=encoder_output,
                tgt_mask=decoder_mask,
                memory_mask=encoder_mask,
                tgt_key_padding_mask=decoder_padding_mask,
                memory_key_padding_mask=encoder_padding_mask,
                past_key_values=past_key_values,
                return_past_key_values=return_past_key_values,
            )

            decoder_output = decoder_outputs.get("decoder_output")
            assert torch.jit.isinstance(decoder_output, Tensor)

            if self.linear_head:
                assert self.lm_head is not None
                # Rescale output before projecting on vocab. This happens when the encoder and decoder share the
                # same word embeddings, which is always the case in our t5 implementation.
                # See https://github.com/huggingface/transformers/blob/d0acc9537829e7d067edbb791473bbceb2ecf056/src/transformers/models/t5/modeling_t5.py#L1661
                decoder_output = decoder_output * (self.embedding_dim ** -0.5)
                decoder_output = self.lm_head(decoder_output)
                decoder_outputs["decoder_output"] = decoder_output

            seq2seq_model_output.update(decoder_outputs)

        return seq2seq_model_output
