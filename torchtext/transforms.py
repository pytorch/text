from . import functional as F
from torch.nn import Module
from torch import Tensor
import torch
from torchtext.data.functional import load_sp_model
from torchtext.utils import get_asset_local_path
from torchtext.vocab import Vocab
from typing import List, Optional, Any, Dict
import json
from functools import lru_cache
import torchtext    # noqa: F401

__all__ = [
    'SentencePieceTokenizer',
    'VocabTransform',
    'ToTensor',
    'LabelToIndex',
    'GPT2BPETokenizer',
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

    def __init__(self, padding_value: Optional[int] = None, dtype: Optional[torch.dtype] = torch.long) -> None:
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


class GPT2BPETokenizer(Module):
    """
    Transform for GPT-2 BPE Tokenizer.

    Reimplements openai GPT-2 BPE in TorchScript. Original openai implementation
    https://github.com/openai/gpt-2/blob/master/src/encoder.py

    :param bpe_encoder: A dict mapping from bpe tokens(e.g sub-words) to
            bpe token ids. For example: {"ingly": 4420}
    :type bpe_encoder: Dict[str, int]
    :param bpe_merge_ranks: A dict mapping a pair of bpe tokens to rank.
            Torchscript doesn't support tuple as dict key type, concat
            a pair of string with SEPERATOR (e.g \u0001) as workaround.
            For example: {"ing\u0001ly": 210}
    :type bpe_merge_ranks: Dict[str, int]
    :param byte_encoder: A dict mapping a byte to an unicode character.
    :type byte_encoder: Dict[int, str]
    """
    SEPERATOR: str = "\u0001"
    LSTRIP_PATTERN: str = "\u0120"

    def __init__(
        self,
        encoder_json_path: str,
        vocab_bpe_path: str,
    ):
        super().__init__()
        self.merge_seperator = self.SEPERATOR
        self.lstrip_pattern = self.LSTRIP_PATTERN
        self.bpe_encoder = self._load_bpe_encoder(encoder_json_path)
        self.bpe_merge_ranks = self._load_bpe_vocab(vocab_bpe_path)
        self.byte_encoder = bytes_to_unicode()

        self.bpe_decoder = self._load_bpe_decoder(self.bpe_encoder)
        self.byte_decoder = self._load_byte_decoder(self.byte_encoder)
        self.inf = len(self.bpe_merge_ranks) + 1

    def _load_bpe_encoder(self, bpe_encoder_path: str) -> Dict[str, int]:
        with open(get_asset_local_path(bpe_encoder_path), "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_bpe_vocab(self, bpe_vocab_path: str) -> Dict[str, int]:
        with open(get_asset_local_path(bpe_vocab_path), "r", encoding="utf-8") as f:
            bpe_vocab = f.read()

        return {
            self.merge_seperator.join(merge_pair.split()): i
            for i, merge_pair in enumerate(bpe_vocab.split("\n")[1:-1])
        }

    def _load_bpe_decoder(self, bpe_encoder: Dict[str, int]) -> Dict[int, str]:
        bpe_decoder: Dict[int, str] = {}
        for sub_word, bpe_token_id in bpe_encoder.items():
            bpe_decoder[bpe_token_id] = sub_word
        return bpe_decoder

    def _load_byte_decoder(self, byte_encoder: Dict[int, str]) -> Dict[str, int]:
        byte_decoder: Dict[str, int] = {}
        for byte_t, unicode_t in byte_encoder.items():
            byte_decoder[unicode_t] = byte_t
        return byte_decoder

    @torch.jit.export
    def _list_str_index(self, list_: List[str], element: str, start: int) -> int:
        """
        Equivalent to: list.index(v, start)
        """
        for i, t in enumerate(list_[start:]):
            if t == element:
                return start + i
        return -1

    @torch.jit.export
    def _get_pairs(self, token_list: List[str]) -> List[str]:
        """Return set of token pairs in a word.

        :param token_list: a list of tokens, each represents a word or sub-word.
        :type input: List[str]
        :rtype: List[str]

        For example: ["he", "l", "l", "o"]
            ==> ["he\u0001l", "l\u0001l", "l\u0001o"]
        """
        pairs: Dict[str, int] = {}
        prev_token: str = token_list[0]
        for token in token_list[1:]:
            pair: str = prev_token + self.merge_seperator + token
            pairs[pair] = 0
            prev_token = token
        return list(pairs.keys())

    @torch.jit.export
    def _find_best_pair(self, pairs: List[str]) -> str:
        """Return the token pair(e.g bpe merge) with lowest rank.

        Equivalent to:
        min(pairs, key = lambda pair: self.bpe_merge_ranks.get(pair, float('inf')))
        """
        best_pair: str = pairs[0]
        best_rank: int = self.bpe_merge_ranks.get(best_pair, self.inf)

        for pair in pairs[1:]:
            rank: int = self.bpe_merge_ranks.get(pair, self.inf)
            if rank < best_rank:
                best_pair = pair
                best_rank = rank
        return best_pair

    @torch.jit.export
    def _bpe(self, token_list: List[str]) -> List[str]:
        """Return a list of bpe tokens.

        Given a list of input tokens, keep finding the best bpe merge and
        generate a new list of tokens until
        1) token list size reduced to 1 or
        2) can't find bpe merge

        :param token_list: a list of encoded bytes, each represented by an unicode.
                For example: ['a', 'w', 'e', 's', 'o', 'm', 'e']
        :type input: List[str]
        :return: A list of bpe tokens generated by greedy search.
                For example: ["aw", "esome"]
        :rtype: List[str]
        """
        pairs: List[str] = self._get_pairs(token_list)

        if len(pairs) == 0:
            return token_list

        while True:
            bigram: str = self._find_best_pair(pairs)
            if bigram not in self.bpe_merge_ranks:
                break

            # Finding all indexes that token_list[i] == first and
            # token_list[i+1] == second.
            # After the loop, new token list will be
            # 1) first + second pair
            # 2) all the other tokens in the original token list
            #
            # For example: first="a" second="w" and token_list =
            # ["a", "w", "some", "a", "w", "e"]
            # Result: new_token_list = ["aw", "some", "aw", "e"]
            first, second = bigram.split(self.merge_seperator)
            new_token_list: List[str] = []
            i: int = 0
            while i < len(token_list):
                j = self._list_str_index(token_list, first, i)
                if j != -1:
                    new_token_list.extend(token_list[i:j])
                    i = j
                else:
                    new_token_list.extend(token_list[i:])
                    break

                if (
                    token_list[i] == first
                    and i < len(token_list) - 1
                    and token_list[i + 1] == second
                ):
                    new_token_list.append(first + second)
                    i += 2
                else:
                    new_token_list.append(token_list[i])
                    i += 1

            token_list = list(new_token_list)
            if len(token_list) == 1:
                break
            else:
                pairs = self._get_pairs(token_list)

        return token_list

    @torch.jit.export
    def _byte_encode(self, token: str) -> List[str]:
        """Encode byte into an unicode character.

        Equivalent to: (self.byte_encoder[b] for b in token.encode('utf-8')
        """
        encoded: List[str] = []
        if torch.jit.is_scripting():
            for b in token:
                encoded.append(self.byte_encoder[ord(b)])
        else:
            encoded = [self.byte_encoder[b] for b in token.encode('utf-8')]
        return encoded

    @torch.jit.export
    def _regex(self, text: str) -> List[str]:
        r"""Return a list of tokens, split by regular expression(pcre).

        's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
        """
        return torch.ops.torchtext.gpt2_bpe_pre_tokenizer(text)

    @torch.jit.export
    def _encode(self, text: str) -> List[int]:
        """Encode text into a list of bpe token ids.

        Split text into a list of token unit, and generate a list of bpe tokens
        for each token unit. Lastly encode bpe tokens into bpe token ids.

        Args:
            text: An input text string.

        Returns:
            A list of bpe token ids represents each bpe tokens.

        For example: "awesome,awe"
            --> tokenize(regex) --> tokens: ["awesome", ",", "awe"]
            --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", e]
            --> bpe encode --> bpe token ids: [707, 5927], [11], [707, 68]
            --> result --> [707, 5927, 11, 707, 68]
        """

        bpe_token_ids: List[int] = []
        for token in self._regex(text):
            for bpe_token in self._bpe(self._byte_encode(token)):
                bpe_token_ids.append(self.bpe_encoder[bpe_token])
        return bpe_token_ids

    @torch.jit.export
    def tokenize(self, text: str) -> List[str]:
        """Encode text into a list of Token(token_id, start_idx, end_idx)

        Args:
            text: An input text string.

        Returns:
            A list of bpe token ids represents each bpe tokens

        For example: "awesome,awe"
            --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", e]
            --> bpe encode --> bpe token ids: [707, 5927, 11, 707, 68]
        """
        bpe_token_ids: List[int] = self._encode(text)
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
                tokens.append(self.tokenize(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            return self.tokenize(input)
        else:
            raise TypeError("Input type not supported")


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
