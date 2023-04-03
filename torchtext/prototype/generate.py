import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

DEFAULT_MAX_SEQ_LEN = 256


class GenerationUtils:
    """Wrapper to provide generation utils for encoder/decoder models and decoder models.

    Example:
    >>> model = T5_BASE_GENERATION.get_model()
    >>> generative_model = GenerationUtils(model=model)
    >>> generative_model.generate(input_ids, num_beams=1, max_length=100)

    The wrapper can work with *any* model as long as it meets the following requirements:
    1. Is an encoder/decoder or decoder based model.
    2. Includes a `get_encoder` method (if applicable) and a `prepare_inputs_for_generation` method.

    This means that popular HuggingFace implementation of T5, Bart, and GPT-2 can all be used with these generation utils!
    >>> from transformers import T5Model
    >>> model = T5Model.from_pretrained("t5-base")
    >>> generative_model = GenerationUtils(model=model, is_huggingface_model=True)
    >>> generative_model.generate(input_ids, num_beams=1, max_length=100)

    `Note`: We cannot make any claims about the stability of APIs from HuggingFace so all models used from the `transformers`
        library are marked 'experimental.'

    More examples can be found in the `notebooks` directory of this repository.
    """

    def __init__(self, model: nn.Module, **kwargs) -> None:
        self.model = model
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", True)
        self.is_huggingface_model = kwargs.pop("is_huggingface_model", False)

    def _prepare_decoder_ids_for_generation(
        self, batch_size: int, pad_idx: int = 0, device: Optional[torch.device] = None, **model_kwargs
    ):
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            return model_kwargs.pop("decoder_input_ids")
        else:
            return torch.ones((batch_size, 1), dtype=torch.long, device=device) * pad_idx

    def greedy_search(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        eos_idx: int,
        pad_idx: int,
        **model_kwargs,
    ) -> torch.Tensor:
        """Greedy search decoding for text generation. Takes the most likely next token every time.

        Inputs:
            input_ids (Tensor): Text prompt(s) for greedy generation.
            max_length (int): Max length to generate responses.
            eos_idx (int): End of sequence index.
            pad_idx (int): Padding index.
            **model_kwargs

        Returns:
            Batch of sequences decoded by greedy search.
        """
        unfinished_sequences = torch.ones((input_ids.shape[0]), device=input_ids.device, dtype=torch.long)

        while True:
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if self.is_huggingface_model:
                model_inputs["return_dict"] = True
                model_inputs["output_hidden_states"] = True

            # Get model output
            outputs = self.model(**model_inputs)
            output_key = "logits" if self.is_huggingface_model else "decoder_output"
            decoder_output = outputs[output_key]

            # Calculate probabilities and take the most likely next token
            probs = F.log_softmax(decoder_output[:, -1], dim=-1)
            next_tokens = torch.argmax(probs, dim=-1)

            # For any finished sequences, padding idx should be the last token
            next_tokens = next_tokens * unfinished_sequences + pad_idx * (1 - unfinished_sequences)

            # Append the next tokens to the previous tokens
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # Update unfinished sequences count
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_idx)).long()

            # Stop iterating once all sequences are finished or exceed the max_length
            if unfinished_sequences.max() == 0 or len(input_ids[0]) >= max_length:
                break

        return input_ids

    def beam_search(self, input_ids: torch.Tensor, num_beams: int, max_length: Optional[int]) -> torch.Tensor:
        raise NotImplementedError()

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        num_beams: Optional[int] = None,
        max_length: Optional[int] = None,
        pad_idx: int = 0,
        eos_idx: int = 1,
    ) -> torch.Tensor:
        """Generation method.

        `num_beams` == 1 or `num_beams` is None -> greedy search
        `num_beams` > 1 -> beam search

        Args:
            input_ids (Tensor): Ids of tokenized input tokens. The 'seed' text for generation.
            num_beams (int): If provided, specifies the number of beams to use in beam search generation.
            max_length (int): Max length to generate responses.
            pad_idx (int): Padding index. Defaults to 0.
            eos_idx (int): End of sequence index. Defaults to 1.
        Returns:
            Tensor of Tensors containing output sequences as ids.

        `Note`: If one beam is provided or no beams are specified, the generation method will default to greedy search.
        """
        model_kwargs = {}

        if self.is_encoder_decoder:
            encoder = self.model.get_encoder()
            encoder_model_kwargs = {"src_key_padding_mask": inputs.eq(pad_idx)}
            model_kwargs["encoder_outputs"] = encoder(inputs, **encoder_model_kwargs)
            inputs = self._prepare_decoder_ids_for_generation(len(inputs), device=inputs.device, **model_kwargs)
            model_kwargs["encoder_padding_mask"] = encoder_model_kwargs.pop("src_key_padding_mask")

        if max_length is None:
            # Too hard to try to figure out the exact max_seq_length for each model
            logger.warning(f"`max_length` was not specified. Defaulting to {DEFAULT_MAX_SEQ_LEN} tokens.")
            max_length = DEFAULT_MAX_SEQ_LEN

        if num_beams == 1 or num_beams is None:
            return self.greedy_search(
                inputs,
                max_length,
                eos_idx,
                pad_idx=pad_idx,
                **model_kwargs,
            )
        elif num_beams > 1:
            return self.beam_search(inputs, num_beams, max_length)
        else:
            raise ValueError("`num_beams` must be >= 1.")

    def _get_top_k_restriction(self, scores: torch.Tensor, top_k: int) -> torch.Tensor:
        """Returns a copy of `scores` restricted to its k highest values (meaning every other value is zeroed)

        Args:
            scores (Tensor): typically the output logits or probabilities for a language model's vocabulary.
            top_k (int): the number of highest values to keep.

        Returns:
            A copy of `scores` restricted to its k highest values
        """
        top_k = min(top_k, scores.size(-1))
        if top_k <= 0:
            raise ValueError(f"`top_k` is {top_k} but should be an int greater than 0")
        indices_to_remove = scores < torch.topk(scores, top_k)[0][:, -1, None]
        return scores.masked_fill(indices_to_remove, 0)

    def _get_top_p_restriction(self, probs: torch.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> torch.Tensor:
        """Returns a copy of `probs` restricted to the top indices whose values sum up to `top_p`
        (meaning the value at any other index is zeroed)

        Args:
            probs (Tensor): output probabilities for a language model vocabulary.
            top_p (float): the (cumulative) threshold for cutting off top-value indices; between 0 and 1.

        Returns:
            A copy of `probs` restricted to the top indices whose values sum up to `top_p`
        """
        if top_p < 0 or top_p > 1:
            raise ValueError(f"`top_p` is {top_p} but should be a float between 0 and 1")
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        sorted_indices_to_keep = cumulative_probs <= top_p
        sorted_indices_to_keep[:, :min_tokens_to_keep] = True
        indices_to_remove = ~sorted_indices_to_keep.scatter(-1, sorted_indices, sorted_indices_to_keep)

        return probs.masked_fill(indices_to_remove, 0)

    def _apply_temperature(self, probs: torch.Tensor, temperature: float) -> torch.Tensor:
        """Applies temperature scaling to `probs`
        Args:
            probs (Tensor): output probabilities for a language model vocabulary.
            temperature (float): value of temperature applied to the distribution.
        Returns:
            A copy of `probs` with applied `temperature`
        """
        if not temperature > 0:
            raise ValueError(f"`temperature` is {temperature} but should be positive")
        return probs / temperature

    def _remove_invalid_values(self, scores: torch.Tensor) -> torch.Tensor:
        """Removes nan and inf values to prevent generation from failing when using sampling"""
        scores[scores != scores] = 0.0
        scores[scores == float("inf")] = torch.finfo(scores.dtype).max
        return scores
