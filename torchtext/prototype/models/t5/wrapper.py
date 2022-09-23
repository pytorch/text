from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchtext.prototype.models import (
    T5_BASE_GENERATION,
    T5_SMALL_GENERATION,
    T5_LARGE_GENERATION,
    T5_3B_GENERATION,
    T5_11B_GENERATION,
    T5Conf,
    T5Transform,
    T5Bundle,
)


BUNDLERS = {
    "base": T5_BASE_GENERATION,
    "small": T5_SMALL_GENERATION,
    "large": T5_LARGE_GENERATION,
    "3b": T5_3B_GENERATION,
    "11b": T5_11B_GENERATION,
}


class T5Wrapper(nn.Module):
    def __init__(
        self,
        configuration: Optional[str] = None,
        checkpoint: Optional[Union[str, Dict[str, torch.Tensor]]] = None,
        t5_config: Optional[T5Conf] = None,
        transform: Optional[T5Transform] = None,
        freeze_model: bool = False,
        strict: bool = False,
        dl_kwargs: Dict[str, Any] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            configuration (str or None): The model configuration. Only support 'base', 'small', 'large', '3b', and '11b' . Must be `None` if checkpoint is not `None`. (Default: `None`)
            checkpoint (str, Dict[str, torch.Tensor], or None): Path to or actual model state_dict. state_dict can have partial weights i.e only for encoder. Must be `None` if configuration is not `None`.(Default: ``None``)
            t5_config (T5Conf or None): An instance of T5Conf that defined the model configuration (i.e. number of layer, attention heads, etc). Must be provided if configuration is `None`. (Default: `None`)
            transform (T5Transfrom or None): An instance of T5Transform that defines the text processing pipeline. Must be provided if configuration is `None`. (Default: `None`)
            freeze_model (bool): Indicates whether to freeze the model weights. (Default: `False`)
            strict (bool): Passed to :func: `torch.nn.Module.load_state_dict` method. (Default: `False`)
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`. (Default: `None`)
        """
        super().__init__()

        if configuration is None:
            assert checkpoint is not None, "Must provide a checkpoint if configuration is None"
            assert t5_config is not None, "Must provide t5_config if using checkpoint"
            assert isinstance(t5_config, T5Conf), f"t5_config must have type {T5Conf.__module__}"
            assert not t5_config.encoder_only, "t5_config.encoder_only must be False"
            assert t5_config.linear_head, "t5_config.linear_head must be True"
            assert transform is not None, "Must provide transform if using checkpoint"
            assert isinstance(transform, T5Transform), f"transform must have type {T5Transform.__module__}"

        else:
            assert checkpoint is None, "configuration and checkpoint were both provided. Can only provide one."
            assert (
                configuration in BUNDLERS
            ), f"Invalid configuration provided. Only support the following configurations: {[key for key in BUNDLERS.keys()]}"

        if configuration is None and checkpoint is not None:
            self.bundler = T5Bundle(_path=checkpoint, _config=t5_config, transform=lambda: transform)
            self.model = self.bundler.build_model(
                config=t5_config, freeze_model=freeze_model, checkpoint=checkpoint, strict=strict, dl_kwargs=dl_kwargs
            )
        else:
            self.bundler = BUNDLERS[configuration]
            self.model = self.bundler.get_model()

        self.transform = self.bundler.transform()

    def beam_search(
        self,
        beam_size: int,
        step: int,
        bsz: int,
        decoder_output: Tensor,
        decoder_tokens: Tensor,
        scores: Tensor,
        incomplete_sentences: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        probs = F.log_softmax(decoder_output[:, -1], dim=-1)
        top = torch.topk(probs, beam_size)

        # N is number of sequences, L is length of sequences, B is beam_size
        # decoder tokens has shape (N,L) -> (N,B,L)
        # top.indices has shape (N,B) - > (N,B,1)
        # x has shape (N,B,L+1)
        # note that when step == 1, N = batch_size, and when step > 1, N = batch_size * beam_size
        x = torch.cat([decoder_tokens.unsqueeze(1).repeat(1, beam_size, 1), top.indices.unsqueeze(-1)], dim=-1)

        # beams are first created for a given sequence
        if step == 1:
            # x has shape (batch_size, B, L+1) -> (batch_size * B,L+1)
            # new_scores has shape (batch_size,B)
            # incomplete_sentences has shape (batch_size * B) = (N)
            new_decoder_tokens = x.view(-1, step + 1)
            new_scores = top.values
            new_incomplete_sentences = incomplete_sentences

        # beams already exist, want to expand each beam into possible new tokens to add
        # and for all expanded beams belonging to the same sequences, choose the top k
        else:
            # scores has shape (batch_size,B) -> (N,1) -> (N,B)
            # top.values has shape (N,B)
            # new_scores has shape (N,B) -> (batch_size, B^2)
            new_scores = (scores.view(-1, 1).repeat(1, beam_size) + top.values).view(bsz, -1)

            # v, i have shapes (batch_size, B)
            v, i = torch.topk(new_scores, beam_size)

            # x has shape (N,B,L+1) -> (batch_size, B, L+1)
            # i has shape (batch_size, B) -> (batch_size, B, L+1)
            # new_decoder_tokens has shape (batch_size, B, L+1) -> (N, L)
            x = x.view(bsz, -1, step + 1)
            new_decoder_tokens = x.gather(index=i.unsqueeze(-1).repeat(1, 1, step + 1), dim=1).view(-1, step + 1)

            # need update incomplete sentences incase one of the beams was kicked out
            # y has shape (N) -> (N, 1) -> (N, B) -> (batch_size, B^2)
            y = incomplete_sentences.unsqueeze(-1).repeat(1, beam_size).view(bsz, -1)

            # now can use i to extract those beams that were selected
            # new_incomplete_sentences has shape (batch_size, B^2) -> (batch_size, B) -> (N, 1) -> N
            new_incomplete_sentences = y.gather(index=i, dim=1).view(bsz * beam_size, 1).squeeze(-1)

            # new_scores has shape (batch_size, B)
            new_scores = v

        return new_decoder_tokens, new_scores, new_incomplete_sentences

    def generate(self, encoder_tokens: Tensor, beam_size: int, eos_idx: int = 1, max_seq_len: int = 512) -> Tensor:

        # pass tokens through encoder
        bsz = encoder_tokens.size(0)
        encoder_padding_mask = encoder_tokens.eq(self.model.padding_idx)
        encoder_embeddings = self.model.dropout1(self.model.token_embeddings(encoder_tokens))
        encoder_output = self.model.encoder(encoder_embeddings, tgt_key_padding_mask=encoder_padding_mask)[0]

        encoder_output = self.model.norm1(encoder_output)
        encoder_output = self.model.dropout2(encoder_output)

        # initialize decoder input sequence; T5 uses padding index as starter index to decoder sequence
        decoder_tokens = torch.ones((bsz, 1), dtype=torch.long) * self.model.padding_idx
        scores = torch.zeros((bsz, beam_size))

        # mask to keep track of sequences for which the decoder has not produced an end-of-sequence token yet
        incomplete_sentences = torch.ones(bsz * beam_size, dtype=torch.long)

        # iteratively generate output sequence until all sequences in the batch have generated the end-of-sequence token
        for step in range(max_seq_len):

            if step == 1:
                # duplicate and order encoder output so that each beam is treated as its own independent sequence
                new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
                new_order = new_order.to(encoder_tokens.device).long()
                encoder_output = encoder_output.index_select(0, new_order)
                encoder_padding_mask = encoder_padding_mask.index_select(0, new_order)

            # causal mask and padding mask for decoder sequence
            tgt_len = decoder_tokens.shape[1]
            decoder_mask = torch.triu(torch.ones((tgt_len, tgt_len), dtype=torch.float64), diagonal=1)
            decoder_mask = decoder_mask.to(torch.bool)
            decoder_padding_mask = decoder_tokens.eq(self.model.padding_idx)

            # T5 implemention uses padding idx to start sequence. Want to ignore this when masking
            decoder_padding_mask[:, 0] = False

            # pass decoder sequence through decoder
            decoder_embeddings = self.model.dropout3(self.model.token_embeddings(decoder_tokens))
            decoder_output = self.model.decoder(
                decoder_embeddings,
                memory=encoder_output,
                tgt_mask=decoder_mask,
                tgt_key_padding_mask=decoder_padding_mask,
                memory_key_padding_mask=encoder_padding_mask,
            )[0]

            decoder_output = self.model.norm2(decoder_output)
            decoder_output = self.model.dropout4(decoder_output)
            decoder_output = decoder_output * (self.model.embedding_dim ** -0.5)
            decoder_output = self.model.lm_head(decoder_output)

            decoder_tokens, scores, incomplete_sentences = self.beam_search(
                beam_size, step + 1, bsz, decoder_output, decoder_tokens, scores, incomplete_sentences
            )
            # ignore newest tokens for sentences that are already complete
            decoder_tokens[:, -1] *= incomplete_sentences

            # update incomplete_sentences to remove those that were just ended
            incomplete_sentences = incomplete_sentences - (decoder_tokens[:, -1] == eos_idx).long()

            if (incomplete_sentences == 0).all():
                break

        # take most likely sequence
        decoder_tokens = decoder_tokens.view(bsz, beam_size, -1)[:, 0, :]
        return decoder_tokens

    def forward(self, input_text: List[str], beam_size: int, max_seq_len: int) -> Union[List[str], str]:

        model_input = self.transform(input_text)
        model_output_tensor = self.generate(encoder_tokens=model_input, beam_size=beam_size, max_seq_len=max_seq_len)
        model_output_list = torch.jit.annotate(List[List[int]], model_output_tensor.tolist())
        output_text = self.transform.decode(model_output_list)

        return output_text
