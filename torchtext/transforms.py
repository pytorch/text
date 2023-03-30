import json
import re
from copy import deepcopy
from functools import lru_cache
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torchtext  # noqa: F401
from torch import Tensor
from torch.nn import Module
from torchtext._torchtext import (
    CLIPEncoder as CLIPEncoderPyBind,
    GPT2BPEEncoder as GPT2BPEEncoderPyBind,
    BERTEncoder as BERTEncoderPyBind,
)
from torchtext._torchtext import RegexTokenizer as RegexTokenizerPybind
from torchtext.data.functional import load_sp_model
from torchtext.utils import get_asset_local_path
from torchtext.vocab import Vocab

from . import functional as F

__all__ = [
    "SentencePieceTokenizer",
    "VocabTransform",
    "ToTensor",
    "LabelToIndex",
    "Truncate",
    "AddToken",
    "PadTransform",
    "StrToIntTransform",
    "GPT2BPETokenizer",
    "CharBPETokenizer",
    "RegexTokenizer",
    "Sequential",
]


class SentencePieceTokenizer(Module):
    """
    Transform for Sentence Piece tokenizer from pre-trained sentencepiece model

    Additional details: https://github.com/google/sentencepiece

    :param sp_model_path: Path to pre-trained sentencepiece model
    :type sp_model_path: str

    Example
        >>> from torchtext.transforms import SentencePieceTokenizer
        >>> transform = SentencePieceTokenizer("spm_model")
        >>> transform(["hello world", "attention is all you need!"])
    """

    def __init__(self, sp_model_path: str) -> None:
        super().__init__()
        self.sp_model = load_sp_model(get_asset_local_path(sp_model_path))

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sentence or list of sentences on which to apply tokenizer.
        :type input: Union[str, List[str]]
        :return: tokenized text
        :rtype: Union[List[str], List[List[str]]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: List[List[str]] = []
            for text in input:
                tokens.append(self.sp_model.EncodeAsPieces(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            return self.sp_model.EncodeAsPieces(input)
        else:
            raise TypeError("Input type not supported")


class VocabTransform(Module):
    r"""Vocab transform to convert input batch of tokens into corresponding token ids

    :param vocab: an instance of :class:`torchtext.vocab.Vocab` class.

    Example:
        >>> import torch
        >>> from torchtext.vocab import vocab
        >>> from torchtext.transforms import VocabTransform
        >>> from collections import OrderedDict
        >>> vocab_obj = vocab(OrderedDict([('a', 1), ('b', 1), ('c', 1)]))
        >>> vocab_transform = VocabTransform(vocab_obj)
        >>> output = vocab_transform([['a','b'],['a','b','c']])
        >>> jit_vocab_transform = torch.jit.script(vocab_transform)
    """

    def __init__(self, vocab: Vocab) -> None:
        super().__init__()
        assert isinstance(vocab, Vocab)
        self.vocab = vocab

    def forward(self, input: Any) -> Any:
        """
        :param input: Input batch of token to convert to correspnding token ids
        :type input: Union[List[str], List[List[str]]]
        :return: Converted input into corresponding token ids
        :rtype: Union[List[int], List[List[int]]]
        """

        if torch.jit.isinstance(input, List[str]):
            return self.vocab.lookup_indices(input)
        elif torch.jit.isinstance(input, List[List[str]]):
            output: List[List[int]] = []
            for tokens in input:
                output.append(self.vocab.lookup_indices(tokens))

            return output
        else:
            raise TypeError("Input type not supported")


class ToTensor(Module):
    r"""Convert input to torch tensor

    :param padding_value: Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    :type padding_value: Optional[int]
    :param dtype: :class:`torch.dtype` of output tensor
    :type dtype: :class:`torch.dtype`
    """

    def __init__(self, padding_value: Optional[int] = None, dtype: torch.dtype = torch.long) -> None:
        super().__init__()
        self.padding_value = padding_value
        self.dtype = dtype

    def forward(self, input: Any) -> Tensor:
        """
        :param input: Sequence or batch of token ids
        :type input: Union[List[int], List[List[int]]]
        :rtype: Tensor
        """
        return F.to_tensor(input, padding_value=self.padding_value, dtype=self.dtype)


class LabelToIndex(Module):
    r"""
    Transform labels from string names to ids.

    :param label_names: a list of unique label names
    :type label_names: Optional[List[str]]
    :param label_path: a path to file containing unique label names containing 1 label per line. Note that either label_names or label_path should be supplied
                       but not both.
    :type label_path: Optional[str]
    """

    def __init__(
        self,
        label_names: Optional[List[str]] = None,
        label_path: Optional[str] = None,
        sort_names=False,
    ) -> None:
        assert label_names or label_path, "label_names or label_path is required"
        assert not (label_names and label_path), "label_names and label_path are mutually exclusive"
        super().__init__()

        if label_path:
            with open(label_path, "r") as f:
                label_names = [line.strip() for line in f if line.strip()]
        else:
            label_names = label_names

        if sort_names:
            label_names = sorted(label_names)
        self._label_vocab = Vocab(torch.classes.torchtext.Vocab(label_names, None))
        self._label_names = self._label_vocab.get_itos()

    def forward(self, input: Any) -> Any:
        """
        :param input: Input labels to convert to corresponding ids
        :type input: Union[str, List[str]]
        :rtype: Union[int, List[int]]
        """
        if torch.jit.isinstance(input, List[str]):
            return self._label_vocab.lookup_indices(input)
        elif torch.jit.isinstance(input, str):
            return self._label_vocab.__getitem__(input)
        else:
            raise TypeError("Input type not supported")

    @property
    def label_names(self) -> List[str]:
        return self._label_names


class Truncate(Module):
    r"""Truncate input sequence

    :param max_seq_len: The maximum allowable length for input sequence
    :type max_seq_len: int
    """

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch of sequence to be truncated
        :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        :return: Truncated sequence
        :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """
        return F.truncate(input, self.max_seq_len)


class AddToken(Module):
    """Add token to beginning or end of sequence

    :param token: The token to be added
    :type token: Union[int, str]
    :param begin: Whether to insert token at start or end or sequence, defaults to True
    :type begin: bool, optional
    """

    def __init__(self, token: Union[int, str], begin: bool = True) -> None:
        super().__init__()
        self.token = token
        self.begin = begin

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch
        :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """

        return F.add_token(input, self.token, self.begin)


class PadTransform(Module):
    """Pad tensor to a fixed length with given padding value.

    :param max_length: Maximum length to pad to
    :type max_length: int
    :param pad_value: Value to pad the tensor with
    :type pad_value: bool
    """

    def __init__(self, max_length: int, pad_value: int) -> None:
        super().__init__()
        self.max_length = max_length
        self.pad_value = float(pad_value)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: The tensor to pad
        :type x: Tensor
        :return: Tensor padded up to max_length with pad_value
        :rtype: Tensor
        """
        max_encoded_length = x.size(-1)
        if max_encoded_length < self.max_length:
            pad_amount = self.max_length - max_encoded_length
            x = torch.nn.functional.pad(x, (0, pad_amount), value=self.pad_value)
        return x


class StrToIntTransform(Module):
    """Convert string tokens to integers (either single sequence or batch)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Any) -> Any:
        """
        :param input: sequence or batch of string tokens to convert
        :type input: Union[List[str], List[List[str]]]
        :return: sequence or batch converted into corresponding token ids
        :rtype: Union[List[int], List[List[int]]]
        """
        return F.str_to_int(input)


class GPT2BPETokenizer(Module):
    """
    Transform for GPT-2 BPE Tokenizer.

    Reimplements openai GPT-2 BPE in TorchScript. Original openai implementation
    https://github.com/openai/gpt-2/blob/master/src/encoder.py

    :param encoder_json_path: Path to GPT-2 BPE encoder json file.
    :type encoder_json_path: str
    :param vocab_bpe_path: Path to bpe vocab file.
    :type vocab_bpe_path: str
    :param return_tokens: Indicate whether to return split tokens. If False, it will return encoded token IDs as strings (default: False)
    :type return_input: bool
    """

    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]
    __jit_unused_properties__ = ["is_jitable"]
    _seperator: torch.jit.Final[str]

    def __init__(self, encoder_json_path: str, vocab_bpe_path: str, return_tokens: bool = False) -> None:
        super().__init__()
        self._seperator = "\u0001"
        # load bpe encoder and bpe decoder
        with open(get_asset_local_path(encoder_json_path), "r", encoding="utf-8") as f:
            bpe_encoder = json.load(f)
        # load bpe vocab
        with open(get_asset_local_path(vocab_bpe_path), "r", encoding="utf-8") as f:
            bpe_vocab = f.read()
        bpe_merge_ranks = {
            self._seperator.join(merge_pair.split()): i for i, merge_pair in enumerate(bpe_vocab.split("\n")[1:-1])
        }
        # Caching is enabled in Eager mode
        self.bpe = GPT2BPEEncoderPyBind(bpe_encoder, bpe_merge_ranks, self._seperator, bytes_to_unicode(), True)

        self._return_tokens = return_tokens

    @property
    def is_jitable(self):
        return isinstance(self.bpe, torch._C.ScriptObject)

    @torch.jit.export
    def _encode(self, text: str) -> List[str]:
        """Encode text into a list of tokens IDs

        Args:
            text: An input text string.

        Returns:
            A list of bpe token ids represents each bpe tokens

        For example: "awesome,awe"
            --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", e]
            --> bpe encode --> bpe token ids: [707, 5927, 11, 707, 68]
        """
        bpe_token_ids: List[int] = self.bpe.encode(text)
        bpe_tokens: List[str] = []

        for bpe_token_id in bpe_token_ids:
            bpe_tokens.append(str(bpe_token_id))

        return bpe_tokens

    @torch.jit.export
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of tokens

        Args:
            text: An input text string.

        Returns:
            A list of bpe token ids represents each bpe tokens

        For example: "awesome,awe"
            --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", e]
        """
        return self.bpe.tokenize(text)

    def add_special_tokens(self, special_tokens_dict: Mapping[str, Union[str, Sequence[str]]]) -> int:
        """Add a dictionary of special tokens (eos, pad, cls…) to the encoder

        :param special_tokens_dict: dict of string. Keys should be in the list of predefined special attributes:
        [bos_token, eos_token, unk_token, sep_token, pad_token, cls_token, mask_token, additional_special_tokens].
        Tokens are only added if they are not already in the vocabulary.
        :type special_tokens_dict: Dict[str, Union[str, List[str]]]
        :return: Number of tokens added to the vocabulary.
        :rtype: int
        """
        for key in special_tokens_dict.keys():
            assert (
                key in self.SPECIAL_TOKENS_ATTRIBUTES
            ), f"Key '{key}' is not in the special token list: {self.SPECIAL_TOKENS_ATTRIBUTES}"

        return self.bpe.add_special_tokens(
            {k: v for k, v in special_tokens_dict.items() if k != "additional_special_tokens"},
            special_tokens_dict.get("additional_special_tokens", []),
        )

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sentence or list of sentences on which to apply tokenizer.
        :type input: Union[str, List[str]]
        :return: tokenized text
        :rtype: Union[List[str], List[List(str)]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: List[List[str]] = []
            for text in input:
                if self._return_tokens:
                    tokens.append(self._tokenize(text))
                else:
                    tokens.append(self._encode(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            if self._return_tokens:
                return self._tokenize(input)
            else:
                return self._encode(input)
        else:
            raise TypeError("Input type not supported")

    def __prepare_scriptable__(self):
        r"""Return a JITable tokenizer."""
        if not self.is_jitable:
            tokenizer_copy = deepcopy(self)
            # Disable caching in script mode
            tokenizer_copy.bpe = torch.classes.torchtext.GPT2BPEEncoder(
                self.bpe.bpe_encoder_, self.bpe.bpe_merge_ranks_, self.bpe.seperator_, self.bpe.byte_encoder_, False
            )
            return tokenizer_copy
        return self

    @torch.jit.export
    def decode(self, tokens: List[str]) -> str:
        """Return a decoded string given a list of string token ids.

        :param input: A list of strings, each string corresponds to token ids.
        :type input: List[str]
        :return: decoded text
        :rtype: str
        """
        return self.bpe.decode([int(token) for token in tokens])


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class CharBPETokenizer(Module):
    """
    Transform for a Character Byte-Pair-Encoding Tokenizer.

    Args:
        :param bpe_encoder_path: Path to the BPE encoder json file.
        :type bpe_encoder_path: str
        :param bpe_merges_path: Path to the BPE merges text file.
        :type bpe_merges_path: str
        :param return_tokens: Indicate whether to return split tokens. If False, it will return encoded token IDs (default: False).
        :type return_tokens: bool
        :param unk_token: The unknown token. If provided, it must exist in encoder.
        :type unk_token: Optional[str]
        :param suffix: The suffix to be used for every subword that is an end-of-word.
        :type suffix: Optional[str]
        :param special_tokens: Special tokens which should not be split into individual characters. If provided, these must exist in encoder.
        :type special_tokens: Optional[List[str]]
    """

    def __init__(
        self,
        bpe_encoder_path: str,
        bpe_merges_path: str,
        return_tokens: bool = False,
        unk_token: Optional[str] = None,
        suffix: Optional[str] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        super().__init__()
        with open(get_asset_local_path(bpe_encoder_path), "r") as f:
            self._encoder = dict(json.load(f))
        with open(get_asset_local_path(bpe_merges_path), "r", encoding="utf-8") as f:
            bpe_data = f.read()

        merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
        self._decoder = {v: k for k, v in self._encoder.items()}
        self._bpe_ranks = dict(zip(merges, range(len(merges))))
        self._return_tokens = return_tokens
        self._cache = {}
        self._pat = r"\S+\n?"
        if unk_token and unk_token not in self._encoder:
            raise RuntimeError(
                "Unknown token {} not found in encoder. Special tokens must be in encoder.".format(unk_token)
            )
        self._unk_token = unk_token
        self._suffix = suffix
        if special_tokens:
            for token in special_tokens:
                if token not in self._encoder:
                    raise RuntimeError(
                        "Special token {} not found in encoder. Special tokens must be in encoder.".format(token)
                    )
                else:
                    self._cache[token] = token

    @property
    def vocab_size(self):
        return len(self._encoder)

    def _bpe(self, token):
        """Splits the input token into bpe tokens. The output depends on the encoder and merge list specified in the class
        constructor. For example, _bpe("pytorch") may return "p y t o r c h" or "py tor ch" or "pytorch" depending on which
        merges exist.

        Args:
            text: An input text string.

        Returns:
            A string of space separated bpe tokens.
        """
        if token in self._cache:
            return self._cache[token]

        if self._suffix:
            word = tuple(token[:-1]) + (token[-1] + self._suffix,)
        else:
            word = tuple(token)

        pairs = get_pairs(word)

        if not pairs:
            if self._suffix:
                return token + self._suffix
            else:
                return token

        while True:
            bigram = min(pairs, key=lambda pair: self._bpe_ranks.get(pair, float("inf")))
            if bigram not in self._bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self._cache[token] = word
        return word

    def encode(self, text: str) -> Union[List[int], List[str]]:
        """Encode text into a list of tokens IDs

        Args:
            text: An input text string.

        Returns:
            A list of bpe token ids represents each bpe token. Return type depends on provided encoder file.
        """
        encoded_tokens = [
            self._encoder.get(bpe_token, self._encoder.get(self._unk_token))
            if self._unk_token
            else self._encoder[bpe_token]
            for bpe_token in self._tokenize(text)
        ]
        return encoded_tokens

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of tokens

        Args:
            text: An input text string.

        Returns:
            A list of bpe token strings
        """
        tokens = []
        for token in re.findall(self._pat, text):
            tokens.extend(bpe_token for bpe_token in self._bpe(token).split(" "))
        return tokens

    def decode(self, tokens: Union[List[int], List[str]]) -> str:
        """Decode a list of token IDs into a string

        Args:
            token: A list of IDs (either str or int depending on encoder json)

        Returns:
            A decoded string
        """
        decoded_list = [
            self._decoder.get(token, self._unk_token) if self._unk_token else self._decoder[token] for token in tokens
        ]
        if self._suffix:
            return "".join(decoded_list).replace(self._suffix, " ")
        else:
            return " ".join(decoded_list)

    def forward(self, input: Union[str, List[str]]) -> Union[List, List[List]]:
        """Forward method of module encodes strings or list of strings into token ids

        Args:
            input: Input sentence or list of sentences on which to apply tokenizer.

        Returns:
            A list or list of lists of token IDs
        """
        if isinstance(input, List):
            tokens: List[List[str]] = []
            for text in input:
                if self._return_tokens:
                    tokens.append(self._tokenize(text))
                else:
                    tokens.append(self.encode(text))
            return tokens
        elif isinstance(input, str):
            if self._return_tokens:
                return self._tokenize(input)
            else:
                return self.encode(input)
        else:
            raise TypeError("Input type not supported")


class CLIPTokenizer(Module):
    """
    Transform for CLIP Tokenizer. Based on Byte-Level BPE.

    Reimplements CLIP Tokenizer in TorchScript. Original implementation:
    https://github.com/mlfoundations/open_clip/blob/main/src/clip/tokenizer.py

    This tokenizer has been trained to treat spaces like parts of the tokens
    (a bit like sentencepiece) so a word will be encoded differently whether it
    is at the beginning of the sentence (without space) or not.

    The below code snippet shows how to use the CLIP tokenizer with encoder and merges file
    taken from the original paper implementation.

    Example
        >>> from torchtext.transforms import CLIPTokenizer
        >>> MERGES_FILE = "http://download.pytorch.org/models/text/clip_merges.bpe"
        >>> ENCODER_FILE = "http://download.pytorch.org/models/text/clip_encoder.json"
        >>> tokenizer = CLIPTokenizer(merges_path=MERGES_FILE, encoder_json_path=ENCODER_FILE)
        >>> tokenizer("the quick brown fox jumped over the lazy dog")

    :param merges_path: Path to bpe merges file.
    :type merges_path: str
    :param encoder_json_path: Optional, path to BPE encoder json file. When specified, this is used
        to infer num_merges.
    :type encoder_json_path: str
    :param num_merges: Optional, number of merges to read from the bpe merges file.
    :type num_merges: int
    :param return_tokens: Indicate whether to return split tokens. If False, it will return encoded token IDs as strings (default: False)
    :type return_input: bool
    """

    __jit_unused_properties__ = ["is_jitable"]
    _seperator: torch.jit.Final[str]

    def __init__(
        self,
        merges_path: str,
        encoder_json_path: Optional[str] = None,
        num_merges: Optional[int] = None,
        return_tokens: bool = False,
    ) -> None:
        super().__init__()
        self._seperator = "\u0001"
        # load bpe merges
        with open(get_asset_local_path(merges_path), "r", encoding="utf-8") as f:
            bpe_merges = f.read().split("\n")[1:]

        if encoder_json_path:
            # load bpe encoder
            with open(get_asset_local_path(encoder_json_path), "r", encoding="utf-8") as f:
                bpe_encoder = json.load(f)
            # 256 * 2 for each byte. For each byte we have ['a', 'a</w>']
            # Additional 2 tokens for bos and eos
            num_merges = len(bpe_encoder) - (256 * 2 + 2)
            bpe_merge_ranks = {
                self._seperator.join(merge_pair.split()): i for i, merge_pair in enumerate(bpe_merges[:num_merges])
            }
        else:
            num_merges = num_merges or len(bpe_merges)
            bpe_merge_ranks = {
                self._seperator.join(merge_pair.split()): i for i, merge_pair in enumerate(bpe_merges[:num_merges])
            }
            bpe_vocab = list(bytes_to_unicode().values())
            bpe_vocab = bpe_vocab + [v + "</w>" for v in bpe_vocab]
            bpe_vocab.extend(["".join(merge_pair.split()) for merge_pair in bpe_merges[:num_merges]])
            bpe_vocab.extend(["<|startoftext|>", "<|endoftext|>"])
            bpe_encoder = {v: i for i, v in enumerate(bpe_vocab)}

        # Caching is enabled in Eager mode
        self.bpe = CLIPEncoderPyBind(bpe_encoder, bpe_merge_ranks, self._seperator, bytes_to_unicode(), True)

        self._return_tokens = return_tokens

    @property
    def is_jitable(self):
        return isinstance(self.bpe, torch._C.ScriptObject)

    @torch.jit.export
    def _encode(self, text: str) -> List[str]:
        """Encode text into a list of tokens IDs

        Args:
            text: An input text string.

        Returns:
            A list of bpe token ids represents each bpe tokens

        For example: "awesome,awe"
            --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", "e"]
            --> bpe encode --> bpe token ids: [707, 5927, 11, 707, 68]
        """
        text = text.lower().strip()
        bpe_token_ids: List[int] = self.bpe.encode(text)
        bpe_tokens: List[str] = []

        for bpe_token_id in bpe_token_ids:
            bpe_tokens.append(str(bpe_token_id))

        return bpe_tokens

    @torch.jit.export
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of tokens

        Args:
            text: An input text string.

        Returns:
            A list of bpe token ids represents each bpe tokens

        For example: "awesome,awe"
            --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", "e"]
        """
        text = text.lower().strip()
        return self.bpe.tokenize(text)

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sentence or list of sentences on which to apply tokenizer.
        :type input: Union[str, List[str]]
        :return: tokenized text
        :rtype: Union[List[str], List[List(str)]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: List[List[str]] = []
            for text in input:
                if self._return_tokens:
                    tokens.append(self._tokenize(text))
                else:
                    tokens.append(self._encode(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            if self._return_tokens:
                return self._tokenize(input)
            else:
                return self._encode(input)
        else:
            raise TypeError("Input type not supported")

    def __prepare_scriptable__(self):
        r"""Return a JITable tokenizer."""
        if not self.is_jitable:
            tokenizer_copy = deepcopy(self)
            # Disable caching in script mode
            tokenizer_copy.bpe = torch.classes.torchtext.CLIPEncoder(
                self.bpe.bpe_encoder_, self.bpe.bpe_merge_ranks_, self.bpe.seperator_, self.bpe.byte_encoder_, False
            )
            return tokenizer_copy
        return self


class BERTTokenizer(Module):
    """
    Transform for BERT Tokenizer.

    Based on WordPiece algorithm introduced in paper:
    https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf

    The backend kernel implementation is taken and modified from https://github.com/LieluoboAi/radish.

    See PR https://github.com/pytorch/text/pull/1707 summary for more details.

    The below code snippet shows how to use the BERT tokenizer using the pre-trained vocab files.

    Example
        >>> from torchtext.transforms import BERTTokenizer
        >>> VOCAB_FILE = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
        >>> tokenizer = BERTTokenizer(vocab_path=VOCAB_FILE, do_lower_case=True, return_tokens=True)
        >>> tokenizer("Hello World, How are you!") # single sentence input
        >>> tokenizer(["Hello World","How are you!"]) # batch input

    :param vocab_path: Path to pre-trained vocabulary file. The path can be either local or URL.
    :type vocab_path: str
    :param do_lower_case: Indicate whether to do lower case. (default: True)
    :type do_lower_case: Optional[bool]
    :param strip_accents: Indicate whether to strip accents. (default: None)
    :type strip_accents: Optional[bool]
    :param return_tokens: Indicate whether to return tokens. If false, returns corresponding token IDs as strings (default: False)
    :type return_tokens: bool
    :param never_split: Collection of tokens which will not be split during tokenization. (default: None)
    :type never_split: Optional[List[str]]
    """

    __jit_unused_properties__ = ["is_jitable"]

    def __init__(
        self,
        vocab_path: str,
        do_lower_case: bool = True,
        strip_accents: Optional[bool] = None,
        return_tokens=False,
        never_split: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        if never_split is None:
            never_split = []
        self.bert_model = BERTEncoderPyBind(
            get_asset_local_path(vocab_path, overwrite=True), do_lower_case, strip_accents, never_split
        )
        self._return_tokens = return_tokens
        self._vocab_path = vocab_path
        self._do_lower_case = do_lower_case
        self._strip_accents = strip_accents
        self._never_split = never_split

    @property
    def is_jitable(self):
        return isinstance(self.bert_model, torch._C.ScriptObject)

    @torch.jit.export
    def _encode(self, text: str) -> List[str]:
        """Encode text into a list of tokens IDs

        Args:
            text: An input text string.

        Returns:
            A list of token ids represents each sub-word

        For example:
            --> "Hello world!" --> token ids: [707, 5927, 11, 707, 68]
        """
        token_ids: List[int] = self.bert_model.encode(text.strip())
        tokens_ids_str: List[str] = [str(token_id) for token_id in token_ids]
        return tokens_ids_str

    @torch.jit.export
    def _batch_encode(self, text: List[str]) -> List[List[str]]:
        """Batch version of _encode i.e operate on list of str"""
        token_ids: List[List[int]] = self.bert_model.batch_encode([t.strip() for t in text])
        tokens_ids_str: List[List[str]] = [[str(t) for t in token_id] for token_id in token_ids]
        return tokens_ids_str

    @torch.jit.export
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of tokens

        Args:
            text: An input text string.

        Returns:
            A list of tokens (sub-words)

        For example:
            --> "Hello World!": ["Hello", "World", "!"]
        """
        return self.bert_model.tokenize(text.strip())

    @torch.jit.export
    def _batch_tokenize(self, text: List[str]) -> List[List[str]]:
        """Batch version of _tokenize i.e operate on list of str"""
        return self.bert_model.batch_tokenize([t.strip() for t in text])

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sentence or list of sentences on which to apply tokenizer.
        :type input: Union[str, List[str]]
        :return: tokenized text
        :rtype: Union[List[str], List[List(str)]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: List[List[str]] = []
            if self._return_tokens:
                tokens = self._batch_tokenize(input)
            else:
                tokens = self._batch_encode(input)
            return tokens
        elif torch.jit.isinstance(input, str):
            if self._return_tokens:
                return self._tokenize(input)
            else:
                return self._encode(input)
        else:
            raise TypeError("Input type not supported")

    def __prepare_scriptable__(self):
        if not self.is_jitable:
            tokenizer_copy = deepcopy(self)
            tokenizer_copy.bert_model = torch.classes.torchtext.BERTEncoder(
                self._vocab_path, self._do_lower_case, self._strip_accents, self._never_split
            )
            return tokenizer_copy

        return self


class RegexTokenizer(Module):
    """
    Regex tokenizer for a string sentence that applies all regex replacements defined in patterns_list. It is backed by the `C++ RE2 regular expression engine <https://github.com/google/re2>`_ from Google.

    Args:
        patterns_list (List[Tuple[str, str]]): a list of tuples (ordered pairs) which contain the regex pattern string
        as the first element and the replacement string as the second element.

    Caveats
        - The RE2 library does not support arbitrary lookahead or lookbehind assertions, nor does it support backreferences. Look at the `docs <https://swtch.com/~rsc/regexp/regexp3.html#caveats>`_ here for more info.
        - The final tokenization step always uses spaces as separators. To split strings based on a specific regex pattern, similar to Python's `re.split <https://docs.python.org/3/library/re.html#re.split>`_, a tuple of ``('<regex_pattern>', ' ')`` can be provided.

    Example
        Regex tokenization based on ``(patterns, replacements)`` list.
            >>> import torch
            >>> from torchtext.transforms import RegexTokenizer
            >>> test_sample = 'Basic Regex Tokenization for a Line of Text'
            >>> patterns_list = [
                (r'\'', ' \'  '),
                (r'\"', '')]
            >>> reg_tokenizer = RegexTokenizer(patterns_list)
            >>> jit_reg_tokenizer = torch.jit.script(reg_tokenizer)
            >>> tokens = jit_reg_tokenizer(test_sample)
        Regex tokenization based on ``(single_pattern, ' ')`` list.
            >>> import torch
            >>> from torchtext.transforms import RegexTokenizer
            >>> test_sample = 'Basic.Regex,Tokenization_for+a..Line,,of  Text'
            >>> patterns_list = [
                (r'[,._+ ]+', r' ')]
            >>> reg_tokenizer = RegexTokenizer(patterns_list)
            >>> jit_reg_tokenizer = torch.jit.script(reg_tokenizer)
            >>> tokens = jit_reg_tokenizer(test_sample)
    """

    __jit_unused_properties__ = ["is_jitable"]

    def __init__(self, patterns_list) -> None:
        super(RegexTokenizer, self).__init__()
        patterns = [pair[0] for pair in patterns_list]
        replacements = [pair[1] for pair in patterns_list]
        self.regex_tokenizer = RegexTokenizerPybind(patterns, replacements, False)

    @property
    def is_jitable(self):
        return not isinstance(self.regex_tokenizer, RegexTokenizerPybind)

    def forward(self, line: str) -> List[str]:
        r"""
        Args:
            lines (str): a text string to tokenize.

        Returns:
            List[str]: a token list after regex.
        """

        return self.regex_tokenizer.forward(line)

    def __prepare_scriptable__(self):
        r"""Return a JITable RegexTokenizer."""

        if not self.is_jitable:
            regex_tokenizer_copy = deepcopy(self)
            regex_tokenizer_copy.regex_tokenizer = torch.classes.torchtext.RegexTokenizer(
                self.regex_tokenizer.patterns_, self.regex_tokenizer.replacements_, False
            )
            return regex_tokenizer_copy

        return self


@lru_cache()
def bytes_to_unicode():
    """
    Original Source: https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9

    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class Sequential(torch.nn.Sequential):
    r"""A container to host a sequence of text transforms."""

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch. The input type must be supported by the first transform in the sequence.
        :type input: `Any`
        """
        for module in self:
            input = module(input)
        return input


class MaskTransform(torch.nn.Module):
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
        >>> from torchtext.transforms import MaskTransform
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
            1) tokens: Tensor with token ids of shape (batch_size x seq_len). Includes token ids for special tokens such as [BOS] and [PAD]
            2) mask_prob: Probability of masking a particular token

        Outputs:
            mask: Tensor with same shape as input tokens (batch_size x seq_len)
                with masked tokens represented by a 1 and everything else as 0.
        """
        batch_size, seq_len = tokens.size()
        num_masked_per_seq = int(seq_len * mask_prob)

        mask = torch.zeros((batch_size, seq_len), dtype=torch.int).to(tokens.device)
        mask[:, :num_masked_per_seq] = 1
        for i in range(batch_size):
            mask[i] = mask[i, torch.randperm(seq_len)]

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
