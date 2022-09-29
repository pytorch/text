from math import inf
from torch import nn
import torch

from flashlight.lib.text.decoder import (
    LexiconSeq2SeqDecoder,
    LexiconFreeSeq2SeqDecoder,
    LexiconFreeSeq2SeqDecoderOptions,
    LexiconSeq2SeqDecoderOptions,
    Trie,
    ZeroLM
)


class GenerationUtil:
    """
    Example:
    >>> t5_base = T5_BASE_GENERATION
    >>> transform = t5_base.transform()
    >>> model = t5_base.get_model()
    >>> model.eval()
    >>> model.to(DEVICE)

    >>> generation_model = GenerationUtil(model)
    >>> inputs = transform(["Hello, it is nice to"])
    >>> generation_model.generate(inputs, num_beams=3, max_len=10)
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def _beam_search(self, input_ids: torch.Tensor, num_beams: int, max_len: int) -> torch.LongTensor:
        options = LexiconFreeSeq2SeqDecoderOptions(
            beam_size=num_beams,
            beam_size_token=self.model.config.vocab_size,
            beam_threshold=50,
            lm_weight=2,
            eos_score=0,
            log_add=False
        )

        import pdb
        pdb.set_trace()

        decoder = LexiconFreeSeq2SeqDecoder(
            options,
            ZeroLM(),
            1,
            None,
            max_output_length=max_len
        )

        output = self.model(input_ids)

        return decoder.decode(output["decoder_output"].data_ptr())

        
    
    @torch.no_grad()
    def generate(self, input_ids, num_beams: int, max_len: int) -> torch.LongTensor:
        print(input_ids)
        if num_beams > 1:
            return self._beam_search(input_ids, num_beams, max_len)
        else:
            return self._greedy_decode()