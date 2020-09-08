from collections import OrderedDict

from fairseq.data.dictionary import Dictionary
import torch
from torchtext.experimental.vocab import vocab, Vocab
from typing import Dict, List, Optional


def build_fairseq_vocab(
    vocab_file: str,
    dictionary_class: Dictionary = Dictionary,
    special_token_replacements: Dict[str, str] = None,
    unk_token: str = "<unk>",
    max_vocab: int = -1,
    min_count: int = -1,
    tokens_to_add: Optional[List[str]] = None,
):
    """Function builds a torchtext Vocab for models pre-trained using Fairseq
    modules.

    The dictionary class can take any Fairseq Dictionary class and is
    used to load the vocab file.

    """
    if not special_token_replacements:
        special_token_replacements = {
            "<pad>": "__PAD__",
            "<s>": "__BEGIN_OF_SENTENCE__",
            "</s>": "__END_OF_SENTENCE__",
            "<unk>": "__UNKNOWN__",
            "<mask>": "__MASK__",
        }
        unk_replacement = special_token_replacements[unk_token] if unk_token in special_token_replacements else unk_token
        special_tokens_to_remove = [special_pair[0] for special_pair in special_token_replacements]
        special_tokens_to_add = tuple(special_pair[1] for special_pair in special_token_replacements if special_pair[0] != unk_token)

    with open(vocab_file) as f:
        dictionary = dictionary_class.load(f)
        # finalize will sort the dict based on frequency so only do this if
        # a min_count or max_vocab size is specified
        if min_count > 0 or max_vocab > 0:
            dictionary.finalize(threshold=min_count, nwords=max_vocab, padding_factor=1)
        if tokens_to_add:
            for token in tokens_to_add:
                dictionary.add_symbol(token)

        dictionary_items = list(zip(dictionary.symbols, dictionary.count))

        ordered_dict = OrderedDict()
        # add special tokens to beginning of ordered_dict
        for s in special_tokens_to_add:
            ordered_dict[s] = 1

        # add all other tokens from dictionary_items
        for token, freq in dictionary_items:
            ordered_dict[token] = freq

        # remove special_tokens_to_remove from dict
        for s in special_tokens_to_remove:
            if s in ordered_dict:
                del ordered_dict[s]

        return Vocab(dictionary_items, unk_token=unk_replacement)


def script_vocab(ordered_dict,
                 pad_token=None,
                 bos_token=None,
                 eos_token=None,
                 mask_token=None,
                 **kwargs):

    v = vocab(ordered_dict, **kwargs)
    return ScriptVocab(v.vocab, pad_token, bos_token, eos_token, mask_token, **kwargs)


class ScriptVocab(Vocab):
    r"""Creates a script vocab object which maps tokens to indices.

    Examples:
        >>> token_to_freq = {'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2}
        >>> sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
        >>> c = OrderedDict(sorted_by_freq_tuples)

        >>> v = ScriptVocab(c)
        >>> v = torch.jit.script(v)

        >>> print(v.lookup_word(0, possible_unk_token='unk'))
        >>> print(v.lookup_indices_1d(['not_present', 'world', 'hello']))
        >>> print(v.lookup_indices_2d([['not_present', 'world', 'hello']]))
        >>> print(v.lookup_words_1d(torch.tensor([0, 1, 2], dtype=torch.int32), [2]))
        >>> print(v.lookup_words_1d_cycle_heuristic(torch.tensor([0, 1, 2, 0], dtype=torch.int32), [2], ['unk_a', 'unk_b']))
        >>> print(v.unk_idx, v.pad_idx, v.bos_idx, v.eos_idx, v.mask_idx)
    """
    def __init__(self,
                 cpp_vocab,
                 pad_token=None,
                 bos_token=None,
                 eos_token=None,
                 mask_token=None,
                 **kwargs):

        super(ScriptVocab, self).__init__(cpp_vocab)

        # store all tokens
        self.unk_token: str = kwargs.get('unk_token', '<unk>')
        self.pad_token: str = pad_token
        self.bos_token: str = bos_token
        self.eos_token: str = eos_token
        self.mask_token: str = mask_token

        # init all special token indices
        self.unk_idx: int = self.vocab[self.unk_token]
        self.pad_idx: int = self.vocab[pad_token] if pad_token and self.vocab[pad_token] != self.unk_idx else -1
        self.bos_idx: int = self.vocab[bos_token] if bos_token and self.vocab[bos_token] != self.unk_idx else -1
        self.eos_idx: int = self.vocab[eos_token] if eos_token and self.vocab[eos_token] != self.unk_idx else -1
        self.mask_idx: int = self.vocab[mask_token] if mask_token and self.vocab[mask_token] != self.unk_idx else -1

    @torch.jit.export
    def lookup_indices_1d(self, values: List[str]) -> List[int]:
        lookup_indices = self.lookup_indices(values)
        return lookup_indices

    @torch.jit.export
    def lookup_indices_2d(self, values: List[List[str]]) -> List[List[int]]:
        result: List[List[int]] = []
        for value in values:
            result.append(self.lookup_indices(value))
        return result

    @torch.jit.export
    def lookup_word(self, idx: int, possible_unk_token: Optional[str] = None) -> str:
        # print(idx, possible_unk_token)
        word = self.lookup_token(idx)
        print(word, self.unk_token)
        if word != self.unk_token or possible_unk_token is None:
            return word
        return possible_unk_token

    @torch.jit.export
    def lookup_words_1d(
        self,
        values: torch.Tensor,
        filter_token_list: List[int] = (),
        possible_unk_token: Optional[str] = None,
    ) -> List[str]:
        """If possible_unk_token is not None, then all UNK id's will be
        replaced by possible_unk_token instead of the default UNK string which
        is <UNK>.

        This is a simple way to resolve UNK's when there's a
        correspondence between source and target translations.

        """
        result: List[str] = []
        for idx in range(values.size(0)):
            value = int(values[idx])
            if value not in filter_token_list:
                token = self.lookup_token(value)
                if token != self.unk_token or possible_unk_token is None:
                    result.append(token)
                else:
                    result.append(possible_unk_token)
        return result

    @torch.jit.export
    def lookup_words_1d_cycle_heuristic(
        self,
        values: torch.Tensor,
        filter_token_list: List[int],
        ordered_unks_token: List[str],
    ) -> List[str]:
        """This function is a extension of the possible_unk_token heuristic in
        lookup_words_1d, which fails in the case when multiple unks are
        available.

        The way we deal with this is we increment every unk token in
        ordered_unks_token everytime we substitute an unk token. This
        solves a substantial amount of queries with multiple unk tokens.

        """
        unk_idx = 0
        unk_idx_length = len(ordered_unks_token)
        vocab_length = len(self.vocab)
        unk_copy = unk_idx_length != 0

        result: List[str] = []
        for idx in range(values.size(0)):
            value = int(values[idx])
            if value not in filter_token_list:
                if value < vocab_length and value != self.unk_idx:
                    result.append(self.lookup_token(value))
                else:
                    if not unk_copy:
                        result.append(self.unk_token)
                    else:
                        unk_value = ordered_unks_token[unk_idx % unk_idx_length]
                        result.append(unk_value)
                        unk_idx += 1
        return result

    def to_ivalue(self):
        r"""Return a JITable ScriptVocab.
        """
        cpp_vocab = torch.classes.torchtext.Vocab(self.vocab.itos_, self.vocab.unk_token_)
        return ScriptVocab(cpp_vocab, self.pad_token, self.bos_token, self.eos_token,
                           self.mask_token, unk_token=self.unk_token)
