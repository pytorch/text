import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from flashlight.lib.text.decoder import (
    LexiconFreeSeq2SeqDecoder,
    LexiconFreeSeq2SeqDecoderOptions,
    ZeroLM,
    create_emitting_model_state,
    get_obj_from_emitting_model_state,
)
from torch import nn


MODEL_KWARGS_TYPE = Dict[str, Dict[str, Union[torch.Tensor, List[Optional[torch.Tensor]], List[torch.Tensor], None]]]

DEFAULT_MAX_SEQ_LEN = 256


@dataclass
class Seq2SeqModelState(object):
    """Seq2SeqModelState for holding state between beam search rounds."""

    timestep: int
    sequence: torch.Tensor
    lm_scores: Optional[torch.Tensor]


class GenerationUtils(nn.Module):
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

    _huggingface_model_input_values = {"return_dict": True, "use_cache": True, "output_hidden_states": True}

    def __init__(self, model: nn.Module, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", True)
        self.is_huggingface_model = kwargs.pop("is_huggingface_model", False)
        self.incremental_decoding = kwargs.pop("incremental_decoding", False)
        if self.is_huggingface_model:
            warnings.warn(
                "PyTorch does not make any claims about the stability of HuggingFace APIs so using all models from the `transformers` library are experimental."
            )

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs: torch.Tensor) -> MODEL_KWARGS_TYPE:
        """Runs encoder and adds to model_kwargs for decoding. Modified from https://github.com/huggingface/transformers/blob/67d074874d285e616393c65a0e670088e1b6b74a/src/transformers/generation/utils.py#L592.

        Args:
            inputs: (Tensor): Tokenized startings sequence(s).
            model_kwargs (Dict[str, Any]): Model keyword arguments to be modified for decoding.

        Returns:
            Modified model_kwargs with addition of encoded input sequence(s).
        """
        # Get encoder
        encoder = self.model.get_encoder()

        # Create copy of encoder kwargs
        encoder_kwargs: Dict[str, bool] = {}

        if self.is_huggingface_model:
            encoder_kwargs["return_dict"] = True

        # Forward pass
        # Explicitly call forward method to assert to assert this is a ScriptModule if JITted
        model_kwargs = {"encoder_outputs": encoder.forward(inputs, **encoder_kwargs)}
        return model_kwargs

    def _prepare_decoder_ids_for_generation(
        self,
        batch_size: int,
        pad_idx: int = 0,
        device: Optional[torch.device] = None,
        model_kwargs: Optional[MODEL_KWARGS_TYPE] = None,
    ) -> torch.Tensor:
        """Prepare decoder IDs for generation."""
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
            assert torch.jit.isinstance(decoder_input_ids, torch.Tensor)
            return decoder_input_ids
        else:
            return torch.ones((batch_size, 1), dtype=torch.long, device=device) * pad_idx

    def greedy_search(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        eos_idx: int,
        pad_idx: Optional[int] = None,
        model_kwargs: Optional[MODEL_KWARGS_TYPE] = {},
    ) -> torch.Tensor:
        """Greedy search decoding for text generation. Takes the most likely next token every time.

        Inputs:
            input_ids (Tensor): Text prompt(s) for greedy generation.
            max_length (int): Max length to generate responses.
            eos_idx (int): End of sequence index.
            pad_idx (int): Padding index.
            model_kwargs

        Returns:
            Batch of sequences decoded by greedy search.
        """
        unfinished_sequences = torch.ones((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.long)

        while True:
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if self.is_huggingface_model:
                model_inputs.update(self._huggingface_model_input_values)

            # Get model output
            outputs = self.model(**model_inputs)
            output_key = "logits" if self.is_huggingface_model else "decoder_output"
            decoder_output = outputs[output_key]

            # Calculate probabilities and take the most likely next token
            probs = F.log_softmax(decoder_output[:, -1], dim=-1)
            _, next_tokens = torch.topk(probs, 1)

            # For any finished sequences, padding idx should be the last token
            if eos_idx is not None and pad_idx is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_idx * (1 - unfinished_sequences)

            # Append the next tokens to the previous tokens
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            if eos_idx is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_idx).long())

            # Stop iterating once all sequences are finished or exceed the max_length
            if unfinished_sequences.max() == 0 or len(input_ids[0]) >= max_length:
                break

        return input_ids

    def beam_search(
        self,
        input_ids: torch.Tensor,
        num_beams: int,
        max_len: int,
        beam_size_token: int,
        beam_threshold: int,
        eos_score: float,
        eos_idx: int,
        num_python_workers: int,
        max_inference_batch_size: int,
        model_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Beam search implemented using Flashlight Text (https://github.com/flashlight/text).

        Args:
            input_ids (Tensor): Tokenized startings sequence(s).
            num_beams (int): Number of beams to use in the beam search.
            max_len (int): Maximum number of tokens to generate.
            beam_size_token (int): Vocab size for the LM being used.
            beam_threshold (int): Threshold before pruning.
            eos_score (float): Score to input when `eos_idx` is generated.
            eos_idx (int): End-of-sequence index.
            num_python_workers (int): Number of python workers to use for multiprocessing.
            model_kwargs

        Returns:
            Tensor of the generated sequences.
        """
        device = input_ids.device

        if self.is_encoder_decoder:
            encoder_output_key = "last_hidden_state" if self.is_huggingface_model else "encoder_output"
            encoder_output = model_kwargs["encoder_outputs"][encoder_output_key]

        def update_func(emissions, N, T, prev_step_token_idxs, prev_step_hyp_idxs, prev_step_model_states, timestep):
            # `emissions` and `N` are unused in this current implementation

            i = T  # Hacky access to the current seq in inputs

            # For first timestep, create previous step token_idxs and model_states
            if timestep == 0:
                prev_step_token_idxs = [-1]
                prev_step_model_states = [
                    create_emitting_model_state(
                        Seq2SeqModelState(timestep=0, sequence=input_ids[i].unsqueeze(0), lm_scores=None)
                    )
                ]

            encoder_output_for_curr_seq = encoder_output[i, :, :].unsqueeze(0) if self.is_encoder_decoder else None
            prev_model_state_sequences = [
                get_obj_from_emitting_model_state(state).sequence for state in prev_step_model_states
            ]
            out_probs, model_states = [], []

            start = 0
            # This is the parallelism level at which elements in the beam will be batched
            step = min(
                max_inference_batch_size, 1000 / (timestep + 1)
            )  # many hypotheses will EOS, so increase the batch size gradually
            curr_beam_size = len(prev_step_token_idxs)

            # 2. Batched inference to get next tokens
            while start < curr_beam_size:  # catch the remainder
                end = start + step
                if end > curr_beam_size:
                    end = curr_beam_size

                num_samples = end - start

                if prev_step_token_idxs != [-1]:
                    state_sequences = torch.cat(prev_model_state_sequences[start:end], dim=0)
                    token_indices = (
                        torch.Tensor(prev_step_token_idxs[start:end])
                        .to(dtype=torch.long, device=device)
                        .reshape(num_samples, 1)
                    )

                    state_and_tokens = torch.cat(
                        [state_sequences, token_indices], dim=-1
                    )  # [batch_size x (timestep + 1)]
                    assert state_and_tokens.shape == (
                        num_samples,
                        timestep + 1,
                    ), f"state_and_tokens has shape {state_and_tokens.shape} = expected {(num_samples, timestep + 1)}"
                else:
                    assert len(prev_model_state_sequences) == 1
                    state_and_tokens = prev_model_state_sequences[0]

                # Cleanup -- combine this with the above
                if self.is_encoder_decoder:
                    # Expand encoder outputs along the batch dimension so that they match the decoder input state's batch size
                    # This is a view-only operation and doesn't copy
                    model_kwargs["encoder_outputs"][encoder_output_key] = encoder_output_for_curr_seq.expand(
                        num_samples if timestep > 0 else 1, -1, -1
                    )

                # Preprocess inputs for generation
                model_inputs = self.model.prepare_inputs_for_generation(state_and_tokens, **model_kwargs)
                if self.is_huggingface_model:
                    model_inputs.update(self._huggingface_model_input_values)

                # Forward pass
                outputs = self.model(**model_inputs)

                # Collect outputs
                output_key = "logits" if self.is_huggingface_model else "decoder_output"
                lm_scores = outputs[output_key]

                # Keep track of probabilities over vocab for this pairing
                for sample_idx in range(num_samples):
                    sample_lm_scores = lm_scores[sample_idx, -1]
                    out_probs.append(sample_lm_scores.tolist())
                    # Keep track of sequence and decoder hidden states
                    model_states.append(
                        create_emitting_model_state(
                            Seq2SeqModelState(
                                timestep=timestep,
                                sequence=state_and_tokens[sample_idx].unsqueeze(0),
                                lm_scores=sample_lm_scores,
                            )
                        )
                    )

                start += step

            return out_probs, model_states

        # 3. Initialize options and decoder from Flashlight Text
        options = LexiconFreeSeq2SeqDecoderOptions(
            beam_size=num_beams,
            beam_size_token=beam_size_token,
            beam_threshold=beam_threshold,
            lm_weight=0.0,  # We have no custom LM so score is zero
            eos_score=eos_score,
            log_add=True,
        )

        decoder = LexiconFreeSeq2SeqDecoder(
            options=options, lm=ZeroLM(), eos_idx=eos_idx, update_func=update_func, max_output_length=max_len
        )

        # 4. Process outputs from beam decoder
        # TODO: This can definitely be optimized
        def beam_decode_step(timestep: int) -> torch.Tensor:
            # Create these as function b/c unnamed functions (lambdas) cause problems w/ MP
            def select_second_elem_in_tuple(tup: Tuple[List[int], float]) -> float:
                return tup[1]

            def is_not_neg_one(elem: int) -> bool:
                return elem != -1

            # Decode step takes ptr to encoder emissions, i, and beam size token
            # but actually these aren't currently being used.
            decoder.decode_step(0, timestep, 0)
            hyps = decoder.get_all_final_hypothesis()

            # Find the best beam
            token_scores = [(hyp.tokens, hyp.score) for hyp in hyps]
            final_tokens = list(filter(is_not_neg_one, max(token_scores, key=select_second_elem_in_tuple)[0]))

            # Have to prepend the input tokens if decoder-only model
            if not self.is_encoder_decoder:
                final_tokens = input_ids[timestep].tolist() + final_tokens

            # Makeshift padding so that we can stack the tensors
            while len(final_tokens) < max_len:
                final_tokens += [0]

            # Convert from list to tensors
            final_tokens_as_tensors = torch.Tensor(final_tokens).to(torch.long)

            return final_tokens_as_tensors

        if num_python_workers > 1:
            warnings.warn("Multiprocessing has not yet been implemented.")

        all_final_tokens = [beam_decode_step(i) for i in range(len(input_ids))]

        # 5. Return top hypotheses for all input sequences
        return torch.stack(all_final_tokens, dim=0)

    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        num_beams: Optional[int] = None,
        max_len: Optional[int] = None,
        pad_idx: int = 0,
        eos_idx: int = 1,
        beam_threshold: int = 100,
        beam_size_token: Optional[int] = None,
        eos_score: float = -1.0,
        num_python_workers: int = 1,
        max_inference_batch_size: int = 16,
    ):
        """Calls self.generate() method."""
        warnings.warn("Forward method simply calls `GenerationUtils.generate()`. Please use generate method directly.")
        return self.generate(
            inputs=inputs,
            num_beams=num_beams,
            max_len=max_len,
            pad_idx=pad_idx,
            eos_idx=eos_idx,
            beam_threshold=beam_threshold,
            beam_size_token=beam_size_token,
            eos_score=eos_score,
            num_python_workers=num_python_workers,
            max_inference_batch_size=max_inference_batch_size,
        )

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        num_beams: Optional[int] = None,
        max_length: Optional[int] = None,
        pad_idx: int = 0,
        eos_idx: int = 1,
        num_python_workers: int = 1,
        beam_threshold: int = 100,
        vocab_size: Optional[int] = None,
        eos_score: float = -1.0,
        max_inference_batch_size: int = 16,
    ) -> torch.Tensor:
        """Entrypoint generation method.

        Args:
            input_ids (Tensor): Ids of tokenized input tokens. The 'seed' text for generation.
            num_beams (int): If provided, specifies the number of beams to use in beam search generation.
            max_length (int): Max length to generate responses.
            pad_idx (int): Padding index. Defaults to 0.
            eos_idx (int): End-of-sequence index. Defaults to 1.
            num_python_workers (int): If > 1, using multiprocessing on CPU.
            vocab_size (int): Vocab size for the beam search algo to evaluate, can typically default to vocab size of the model.
            beam_threshold (int): Threshold before pruning; specific to beam search.
            eos_score (float): Score to input when `eos_idx` is generated; specific to beam search.
            max_inference_batch_size (int): In beam search, to avoid OOMs, can choose to batch smaller amounts of hypothesis; defaults to 16.

        Returns:
            Tensor of Tensors containing output sequences as ids.

        Conditions for generation:
            1. `num_beams` == 1 or `num_beams` is None -> greedy search
            2. `num_beams` > 1 -> beam search
        """
        model_kwargs: MODEL_KWARGS_TYPE = {}

        if self.is_encoder_decoder:
            assert torch.jit.isinstance(inputs, torch.Tensor)
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(inputs)
            inputs = self._prepare_decoder_ids_for_generation(
                len(inputs), device=inputs.device, model_kwargs=model_kwargs
            )

        if max_length is None:
            # Too hard to try to figure out the exact max_seq_length for each model
            warnings.warn("`max_length` was not specified. Defaulting to 256 tokens.")
            max_length = 256

        if num_beams is None or num_beams == 1:
            if num_python_workers > 1:
                warnings.warn(f"Multiprocessing is not implemented for greedy search.")
            return self.greedy_search(inputs, max_length, eos_idx, pad_idx=pad_idx, model_kwargs=model_kwargs)
        elif num_beams > 1:
            assert (
                vocab_size is not None
            ), "`vocab_size` must be specified for beam search. If confused about what to put, you can default to the vocab size of the model you are using."
            return self.beam_search(
                inputs,
                num_beams,
                beam_size_token=vocab_size,
                max_len=max_length,
                beam_threshold=beam_threshold,
                eos_score=eos_score,
                num_python_workers=num_python_workers,
                eos_idx=eos_idx,
                max_inference_batch_size=max_inference_batch_size,
                model_kwargs=model_kwargs,
            )
        else:
            raise ValueError("`num_beams` must be >= 1.")
