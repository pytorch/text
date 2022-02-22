from . import functional as F
from torch.nn import Module
from torch import Tensor
import torch
from torchtext.data.functional import load_sp_model
from torchtext.utils import get_asset_local_path
from torchtext.vocab import Vocab
from torchtext._torchtext import GPT2BPEEncoder as GPT2BPEEncoderPyBind, CLIPEncoder as CLIPEncoderPyBind
from typing import List, Optional, Any, Union
import json
from functools import lru_cache
from copy import deepcopy
import torchtext    # noqa: F401

__all__ = [
    'SentencePieceTokenizer',
    'VocabTransform',
    'ToTensor',
    'LabelToIndex',
    'Truncate',
    'AddToken',
    'GPT2BPETokenizer',
    'Sequential',
]


class SentencePieceTokenizer(Module):
    """
    Transform for Sentence Piece tokenizer from pre-trained sentencepiece model

    Additiona details: https://github.com/google/sentencepiece

    :param sp_model_path: Path to pre-trained sentencepiece model
    :type sp_model_path: str

    Example
        >>> from torchtext.transforms import SpmTokenizerTransform
        >>> transform = SentencePieceTokenizer("spm_model")
        >>> transform(["hello world", "attention is all you need!"])
    """

    def __init__(self, sp_model_path: str):
        super().__init__()
        self.sp_model = load_sp_model(get_asset_local_path(sp_model_path))

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

    def __init__(self, vocab: Vocab):
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
        self, label_names: Optional[List[str]] = None, label_path: Optional[str] = None, sort_names=False,
    ):
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


class GPT2BPETokenizer(Module):
    __jit_unused_properties__ = ["is_jitable"]
    """
    Transform for GPT-2 BPE Tokenizer.

    Reimplements openai GPT-2 BPE in TorchScript. Original openai implementation
    https://github.com/openai/gpt-2/blob/master/src/encoder.py

    :param encoder_json_path: Path to GPT-2 BPE encoder json file.
    :type encoder_json_path: str
    :param vocab_bpe_path: Path to bpe vocab file.
    :type vocab_bpe_path: str
    """
    _seperator: torch.jit.Final[str]

    def __init__(
        self,
        encoder_json_path: str,
        vocab_bpe_path: str,
    ):
        super().__init__()
        self._seperator = "\u0001"
        # load bpe encoder and bpe decoder
        with open(get_asset_local_path(encoder_json_path), "r", encoding="utf-8") as f:
            bpe_encoder = json.load(f)
        # load bpe vocab
        with open(get_asset_local_path(vocab_bpe_path), "r", encoding="utf-8") as f:
            bpe_vocab = f.read()
        bpe_merge_ranks = {
            self._seperator.join(merge_pair.split()): i
            for i, merge_pair in enumerate(bpe_vocab.split("\n")[1:-1])
        }
        # Caching is enabled in Eager mode
        self.bpe = GPT2BPEEncoderPyBind(bpe_encoder, bpe_merge_ranks,
                                        self._seperator, bytes_to_unicode(), True)

    @property
    def is_jitable(self):
        return isinstance(self.bpe, torch._C.ScriptObject)

    @torch.jit.export
    def _tokenize(self, text: str) -> List[str]:
        """Encode text into a list of tokens

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
                tokens.append(self._tokenize(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            return self._tokenize(input)
        else:
            raise TypeError("Input type not supported")

    def __prepare_scriptable__(self):
        r"""Return a JITable tokenizer.
        """
        if not self.is_jitable:
            tokenizer_copy = deepcopy(self)
            # Disable caching in script mode
            tokenizer_copy.bpe = torch.classes.torchtext.GPT2BPEEncoder(self.bpe.bpe_encoder_,
                                                                        self.bpe.bpe_merge_ranks_,
                                                                        self.bpe.seperator_,
                                                                        self.bpe.byte_encoder_, False)
            return tokenizer_copy
        return self


class CLIPTokenizer(Module):
    __jit_unused_properties__ = ["is_jitable"]
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
    """

    _seperator: torch.jit.Final[str]

    def __init__(self, merges_path: str, encoder_json_path: Optional[str] = None, num_merges: Optional[int] = None):
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
        self.bpe = CLIPEncoderPyBind(bpe_encoder, bpe_merge_ranks,
                                     self._seperator, bytes_to_unicode(), True)

    @property
    def is_jitable(self):
        return isinstance(self.bpe, torch._C.ScriptObject)

    @torch.jit.export
    def _tokenize(self, text: str) -> List[str]:
        """Encode text into a list of tokens

        Args:
            text: An input text string.

        Returns:
            A list of bpe token ids represents each bpe tokens

        For example: "awesome,awe"
            --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", e]
            --> bpe encode --> bpe token ids: [707, 5927, 11, 707, 68]
        """
        text = text.lower().strip()
        bpe_token_ids: List[int] = self.bpe.encode(text)
        bpe_tokens: List[str] = []

        for bpe_token_id in bpe_token_ids:
            bpe_tokens.append(str(bpe_token_id))

        return bpe_tokens

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
                tokens.append(self._tokenize(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            return self._tokenize(input)
        else:
            raise TypeError("Input type not supported")

    def __prepare_scriptable__(self):
        r"""Return a JITable tokenizer.
        """
        if not self.is_jitable:
            tokenizer_copy = deepcopy(self)
            # Disable caching in script mode
            tokenizer_copy.bpe = torch.classes.torchtext.CLIPEncoder(self.bpe.bpe_encoder_,
                                                                     self.bpe.bpe_merge_ranks_,
                                                                     self.bpe.seperator_,
                                                                     self.bpe.byte_encoder_, False)
            return tokenizer_copy
        return self


@lru_cache()
def bytes_to_unicode():
    """
    Original Source: https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9

    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
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
    r"""A container to host a sequence of text transforms.
    """

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch. The input type must be supported by the first transform in the sequence.
        :type input: `Any`
        """
        for module in self:
            input = module(input)
        return input
