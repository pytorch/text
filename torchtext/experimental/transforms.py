import io
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchtext._torchtext import RegexTokenizer as RegexTokenizerPybind, SentencePiece as SentencePiecePybind


__all__ = [
    "basic_english_normalize",
    "BasicEnglishNormalize",
    "PRETRAINED_SP_MODEL",
    "load_sp_model",
    "sentencepiece_tokenizer",
    "SentencePieceTokenizer",
    "sentencepiece_processor",
    "SentencePieceProcessor",
    "VocabTransform",
    "VectorTransform",
]


def basic_english_normalize():
    r"""Basic normalization for a string sentence.

    Normalization includes
        - lowercasing
        - complete some basic text normalization for English words as follows:

            - add spaces before and after '\''
            - remove '\"',
            - add spaces before and after '.'
            - replace '<br \/>'with single space
            - add spaces before and after ','
            - add spaces before and after '('
            - add spaces before and after ')'
            - add spaces before and after '!'
            - add spaces before and after '?'
            - replace ';' with single space
            - replace ':' with single space
            - replace multiple spaces with single space

    Examples:
        >>> import torch
        >>> from torchtext.experimental.transforms import basic_english_normalize
        >>> test_sample = 'Basic English Normalization for a Line of Text'
        >>> basic_eng_norm = basic_english_normalize()
        >>> jit_basic_eng_norm = torch.jit.script(basic_eng_norm)
        >>> tokens = jit_basic_eng_norm(test_sample)
    """

    patterns_list = [
        (r"\'", " '  "),
        (r"\"", ""),
        (r"\.", " . "),
        (r"<br \/>", " "),
        (r",", " , "),
        (r"\(", " ( "),
        (r"\)", " ) "),
        (r"\!", " ! "),
        (r"\?", " ? "),
        (r"\;", " "),
        (r"\:", " "),
        (r"\s+", " "),
    ]

    patterns = [pair[0] for pair in patterns_list]
    replacements = [pair[1] for pair in patterns_list]
    return BasicEnglishNormalize(RegexTokenizerPybind(patterns, replacements, True))


class BasicEnglishNormalize(nn.Module):
    __jit_unused_properties__ = ["is_jitable"]
    r"""Basic normalization for a string sentence.

    Args:
        regex_tokenizer (torch.classes.torchtext.RegexTokenizer or torchtext._torchtext.RegexTokenizer): a cpp regex tokenizer object.
    """

    def __init__(self, regex_tokenizer):
        super(BasicEnglishNormalize, self).__init__()
        self.regex_tokenizer = regex_tokenizer

    @property
    def is_jitable(self):
        return not isinstance(self.regex_tokenizer, RegexTokenizerPybind)

    def forward(self, line: str) -> List[str]:
        r"""
        Args:
            lines (str): a text string to tokenize.

        Returns:
            List[str]: a token list after normalizing and splitting on whitespace.
        """

        return self.regex_tokenizer.forward(line)

    def __prepare_scriptable__(self):
        r"""Return a JITable BasicEnglishNormalize."""
        regex_tokenizer = torch.classes.torchtext.RegexTokenizer(
            self.regex_tokenizer.patterns_, self.regex_tokenizer.replacements_, True
        )
        return BasicEnglishNormalize(regex_tokenizer)


PRETRAINED_SP_MODEL = {
    "text_unigram_15000": "https://pytorch.s3.amazonaws.com/models/text/pretrained_spm/text_unigram_15000.model",
    "text_unigram_25000": "https://pytorch.s3.amazonaws.com/models/text/pretrained_spm/text_unigram_25000.model",
    "text_unigram_50000": "https://pytorch.s3.amazonaws.com/models/text/pretrained_spm/text_unigram_50000.model",
    "text_bpe_15000": "https://pytorch.s3.amazonaws.com/models/text/pretrained_spm/text_bpe_15000.model",
    "text_bpe_25000": "https://pytorch.s3.amazonaws.com/models/text/pretrained_spm/text_bpe_25000.model",
    "text_bpe_50000": "https://pytorch.s3.amazonaws.com/models/text/pretrained_spm/text_bpe_50000.model",
}


def load_sp_model(sp_model):
    r"""Load a  sentencepiece model for file.

    Args:
        sp_model: the file path or a file object saving the sentencepiece model.

    Outputs:
        output: a SentencePiece model.

    Examples:
        >>> from torchtext.experimental.transforms import load_sp_model
        >>> sp_model = load_sp_model("m_user.model")
        >>> sp_model = load_sp_model(open("m_user.model", 'rb'))

    Note: We also provide several pretrained sentencepiece models. The model was trained with torchtext.datasets.WikiText103,
        torchtext.datasets.EnWik9 and BookCorpus. Both BPE and unigram methods were used to train the model (for more
        details please refer to SentencePiece GitHub https://github.com/google/sentencepiece). We also provide the pretrained model
        with a different size of the vocabulary (i.e. 15000, 25000, 50000).
        The following pretrained sentencepiece models are provided:

            - text_unigram_15000
            - text_unigram_25000
            - text_unigram_50000
            - text_bpe_15000
            - text_bpe_25000
            - text_bpe_50000

    Examples:
        >>> from torchtext.experimental.transforms import PRETRAINED_SP_MODEL
        >>> sp_model_path = torchtext.utils.download_from_url(PRETRAINED_SP_MODEL['text_unigram_25000'])
        >>> sp_model = load_sp_model(sp_model_path)
    """

    if isinstance(sp_model, str):
        with open(sp_model, "rb") as f:
            return SentencePiecePybind(f.read())
    elif isinstance(sp_model, io.BufferedReader):
        return SentencePiecePybind(sp_model.read())
    else:
        raise TypeError(
            f"Unsupported type for sp_model argument: {type(sp_model).__name__}. "
            + "Supported types are: "
            + ", ".join(["str", "io.BufferedReader"])
        )


def sentencepiece_tokenizer(sp_model):
    r"""Factory function to generate SentencePieceTokenizer from a pretrained SentencePiece model

    Args:
        sp_model: the file path or a file object saving the sentencepiece model.

    Examples:
        >>> import torch
        >>> from torchtext.experimental.transforms import sentencepiece_tokenizer
        >>> spm_tokenizer = sentencepiece_tokenizer('m_user.model')
        >>> jit_spm_tokenizer = torch.jit.script(spm_tokenizer)
    """

    spm = load_sp_model(sp_model)
    return SentencePieceTokenizer(spm)


class SentencePieceTokenizer(nn.Module):
    r"""Tokenizer based on a pretained sentencepiece model.

    Args:
       spm_model: the sentencepiece model instance
    """

    def __init__(self, spm_model):
        super(SentencePieceTokenizer, self).__init__()
        self.sp_model = spm_model

    def forward(self, line: str) -> List[str]:
        r"""
        Args:
            line: the input sentence string

        Examples:
            >>> spm_tokenizer('the pretrained sp model names')
            >>> ['▁the', '▁pre', 'trained', '▁sp', '▁model', '▁names']

        Note: SentencePiece treats the input text just as a sequence of Unicode characters. Whitespace is also handled as a normal symbol. To handle the whitespace as a basic token explicitly, SentencePiece first escapes the whitespace with a meta symbol "▁" (U+2581) as follows.
        """

        return self.sp_model.EncodeAsPieces(line)

    @torch.jit.export
    def decode(self, tokens: List[str]) -> str:
        r"""
        Args:
            tokens: the tokens list for decoder

        Examples:
            >>> spm_transform.decoder(['▁the', '▁pre', 'trained', '▁sp', '▁model', '▁names'])
            >>> 'the pretrained sp model names'
        """

        return self.sp_model.DecodePieces(tokens)

    def __prepare_scriptable__(self):
        torchbind_spm = torch.classes.torchtext.SentencePiece(self.sp_model._return_content())
        return SentencePieceTokenizer(torchbind_spm)


def sentencepiece_processor(sp_model):
    r"""Factory function to generate SentencePieceProcessor from a pretrained SentencePiece model

    Args:
        sp_model: the file path or a file object saving the sentencepiece model.

    Examples:
        >>> import torch
        >>> from torchtext.experimental.transforms import sentencepiece_processor
        >>> spm_processor = sentencepiece_processor('m_user.model')
        >>> jit_spm_processor = torch.jit.script(spm_processor)
    """

    spm = load_sp_model(sp_model)
    return SentencePieceProcessor(spm)


class SentencePieceProcessor(nn.Module):
    r"""String to ids transform based on a pretained sentencepiece model

    Args:
       spm_model: the sentencepiece model instance
    """

    def __init__(self, spm_model):
        super(SentencePieceProcessor, self).__init__()
        self.sp_model = spm_model

    def forward(self, line: str) -> List[int]:
        r"""
        Args:
            line: the input sentence string

        Examples:
            >>> spm_processor('the pretrained sp model names')
            >>> [9, 1546, 18811, 2849, 2759, 2202]
        """

        return self.sp_model.EncodeAsIds(line)

    @torch.jit.export
    def decode(self, ids: List[int]) -> str:
        r"""
        Args:
            ids: the integers list for decoder

        Examples:
            >>> spm_processor.decoder([9, 1546, 18811, 2849, 2759, 2202])
            >>> 'the pretrained sp model names'
        """

        return self.sp_model.DecodeIds(ids)

    def __prepare_scriptable__(self):
        torchbind_spm = torch.classes.torchtext.SentencePiece(self.sp_model._return_content())
        return SentencePieceProcessor(torchbind_spm)


class VocabTransform(nn.Module):
    r"""Vocab transform

    Args:
        vocab: an instance of torchtext.vocab.Vocab class.

    Example:
        >>> import torch
        >>> from torchtext.vocab import vocab_from_file_object
        >>> f = open('vocab.txt', 'r')
        >>> vocab_transform = VocabTransform(vocab_from_file_object(f))
        >>> jit_vocab_transform = torch.jit.script(vocab_transform)
    """

    def __init__(self, vocab):
        super(VocabTransform, self).__init__()
        self.vocab = vocab

    def forward(self, tokens: List[str]) -> List[int]:
        r"""

        Args:
            tokens: a string token list

        Example:
            >>> vocab_transform(['here', 'is', 'an', 'example'])

        """

        return self.vocab.lookup_indices(tokens)


class VectorTransform(nn.Module):
    r"""Vector transform

    Args:
        vector: an instance of torchtext.experimental.vectors.Vectors class.

    Example:
        >>> import torch
        >>> from torchtext.experimental.vectors import FastText
        >>> vector_transform = VectorTransform(FastText())
        >>> jit_vector_transform = torch.jit.script(vector_transform)
    """

    def __init__(self, vector):
        super(VectorTransform, self).__init__()
        self.vector = vector

    def forward(self, tokens: List[str]) -> Tensor:
        r"""

        Args:
            tokens: a string token list

        Example:
            >>> vector_transform(['here', 'is', 'an', 'example'])

        """

        return self.vector.lookup_vectors(tokens)


class MaskTransform(nn.Module):
    """
    The transform chooses mask_prob% (example 15%) of the token positions at random for
    prediction.

    If the i-th token is chosen, we replace the i-th token with
    (1) the [MASK] token 80% of the time
    (2) a random token 10% of the time
    (3) the unchanged i-th token 10% of the time.

    Args:
        vocab_len (int): the length of the vocabulary, including special tokens such as [BOS], [PAD], [MASK]
        mask_idx (int): index assigned to mask token in vocabulary
        bos_idx (int): index assigned to beginning-of-sequence token in vocabulary
        pad_idx (int): index assigned to padding token in vocabulary
        mask_bos (bool): indicate whether beginning-of-sequence tokens are eligible for masking (default: False)
        mask_prob (float): probability that a token is chosen for replacement (default: 0.15)

    Example:
        >>> import torch
        >>> from torchtext.experimental.transforms import MaskTransform
        >>> sample_tokens = [
                ["[BOS]", "a", "b", "c", "d"],
                ["[BOS]", "a", "b", "[PAD]", "[PAD]"]
            ]
        >>> sample_token_ids = torch.tensor([
                [6, 0, 1, 2, 3], [6, 0, 1, 4, 4]
            ])
        >>> mask_transform = MaskTransform(
                vocab_len = 7,
                mask_idx = 4,
                bos_idx = 6,
                pad_idx = 5,
                mask_bos = False,
                mask_prob = 0.15
            )
        >>> masked_tokens, target_tokens, mask = mask_transform(sample_token_ids)
    """

    # maks_mask_prob is prob. of replacing a token with [MASK] (ex. 80%)
    mask_mask_prob = 0.8

    # rand_mask_thresh is prob. of replacing a token with a random token. (ex.10%)
    rand_mask_prob = 0.1

    def __init__(
        self,
        vocab_len: int,
        mask_idx: int,
        bos_idx: int,
        pad_idx: int,
        mask_bos: bool = False,
        mask_prob: float = 0.15,
    ):
        super().__init__()
        self.vocab_len = vocab_len
        self.mask_idx = mask_idx
        self.bos_idx = bos_idx
        self.pad_idx = pad_idx
        self.mask_prob = mask_prob
        self.mask_bos = mask_bos

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies mask to input tokens.

        Inputs:
            tokens: Tensor with token ids of shape (batch_size x seq_len). Includes token ids for special tokens such as [BOS] and [PAD]

        Outputs:
            masked_tokens: Tensor of tokens after masking has been applied
            target_tokens: Tensor of token values selected for masking
            mask: Tensor with same shape as input tokens (batch_size x seq_len)
                with masked tokens represented by a 1 and everything else as 0.
        """
        # tokens, mask, mask_mask, rand_mask: (T, C)
        mask, mask_mask, rand_mask = self._generate_mask(tokens)

        # a. generate the masked input tokens
        # (1) the [MASK] token 80% of the time
        masked_tokens = self._mask_input(tokens, mask_mask, self.mask_idx)
        # (2) a random token 10% of the time
        masked_tokens = self._mask_input(
            masked_tokens,
            rand_mask,
            torch.randint_like(tokens, high=self.vocab_len),
        )

        # b. generate the target prediction
        target_tokens = torch.masked_select(tokens, mask.bool())

        # masked_tokens: (T, C), target_tokens: (T x C x mask_prob, ), mask
        return masked_tokens, target_tokens, mask

    def _random_masking(self, tokens: torch.tensor, mask_prob: float) -> torch.Tensor:
        """
        Function to mask tokens randomly.

        Inputs:
            1) tokens: Tensor with token ids of shape (batch_size x seq_len)
            2) mask_prob: Probability of masking a particular token

        Outputs:
            mask: Tensor with same shape as input tokens (batch_size x seq_len)
                with masked tokens represented by a 1 and everything else as 0.
        """
        batch_size, seq_len = tokens.size()
        num_masked_per_seq = int(seq_len * mask_prob)

        mask = np.zeros((batch_size, seq_len), dtype=np.int_)
        mask[:, :num_masked_per_seq] = 1
        for row in mask:
            np.random.shuffle(row)
        mask = torch.from_numpy(mask).to(tokens.device)
        return mask

    def _select_tokens_to_mask(self, tokens: torch.Tensor, mask_prob: float) -> torch.Tensor:
        mask = self._random_masking(tokens, mask_prob)
        if not self.mask_bos:
            mask *= (tokens != self.bos_idx).long()
        return mask

    def _generate_mask(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # chooses mask_prob% of the token positions at random
        mask = self._select_tokens_to_mask(tokens, self.mask_prob)
        # not mask the pad token
        mask *= (tokens != self.pad_idx).long()
        # keep one masked token to avoid failure in the loss calculation.
        mask[0, 0] = 1 if not mask.byte().any() else mask[0, 0]

        probs = torch.rand_like(tokens, dtype=torch.float)
        # (1) the [MASK] token 80% of the time
        mask_mask = (probs >= (1 - self.mask_mask_prob)).long() * mask
        # (2) a random token 10% of the time
        rand_mask = (probs < self.rand_mask_prob).long() * mask
        return mask, mask_mask, rand_mask

    def _mask_input(self, tokens: torch.Tensor, mask: torch.Tensor, replacement) -> torch.Tensor:
        return tokens * (1 - mask) + replacement * mask
