from collections import OrderedDict

# from fairseq.data.dictionary import Dictionary
import torch
from torchtext.experimental.vocab import Vocab, create_vocab_factory
from typing import List, Optional, Set


# def build_fairseq_vocab(
#     vocab_file: str,
#     dictionary_class: Dictionary = Dictionary,
#     special_token_replacements: Dict[str, str] = None,
#     unk_token: str = "<unk>",
#     max_vocab: int = -1,
#     min_count: int = -1,
#     tokens_to_add: Optional[List[str]] = None,
# ):
#     """Function builds a torchtext Vocab for models pre-trained using Fairseq
#     modules.

#     The dictionary class can take any Fairseq Dictionary class and is
#     used to load the vocab file.

#     """
#     if not special_token_replacements:
#         special_token_replacements = {
#             "<pad>": "__PAD__",
#             "<s>": "__BEGIN_OF_SENTENCE__",
#             "</s>": "__END_OF_SENTENCE__",
#             "<unk>": "__UNKNOWN__",
#             "<mask>": "__MASK__",
#         }
#         unk_replacement = special_token_replacements[unk_token] if unk_token in special_token_replacements else unk_token
#         special_tokens_to_remove = [special_pair[0] for special_pair in special_token_replacements]
#         specials = tuple(special_pair[1] for special_pair in special_token_replacements if special_pair[0] != unk_token)

#     with open(vocab_file) as f:
#         dictionary = dictionary_class.load(f)
#         # finalize will sort the dict based on frequency so only do this if
#         # a min_count or max_vocab size is specified
#         if min_count > 0 or max_vocab > 0:
#             dictionary.finalize(threshold=min_count, nwords=max_vocab, padding_factor=1)
#         if tokens_to_add:
#             for token in tokens_to_add:
#                 dictionary.add_symbol(token)

#         dictionary_items = list(zip(dictionary.symbols, dictionary.count))
#         ordered_dict = OrderedDict(dictionary_items)

#         # remove specials from dict since Vocab expects a seperate tuple of special tokens
#         for s in special_tokens_to_remove:
#             if s in ordered_dict:
#                 del ordered_dict[s]

#         return Vocab(dictionary_items, unk_token=unk_replacement, specials=specials)


class ScriptVocab(Vocab):
    def __init__(self,
                 ordered_dict,
                 pad_token=None,
                 bos_token=None,
                 eos_token=None,
                 mask_token=None,
                 **kwargs):
        # super(ScriptVocab, self).__init__(ordered_dict, **kwargs)
        # super(ScriptVocab, self).__init__(create_vocab_factory(ordered_dict, **kwargs).vocab)
        vocab = create_vocab_factory(ordered_dict, **kwargs)
        vocab_cpp = vocab.vocab
        # import pdb
        # pdb.set_trace()
        super(ScriptVocab, self).__init__(vocab_cpp)
        self.unk_token: str = kwargs.get('unk_token', '<unk>')

        # init all special token indices
        self.unk_idx: int = self.vocab[self.unk_token]
        self.pad_idx: int = self.vocab[pad_token] if pad_token and self.vocab[pad_token] != self.unk_idx else -1
        self.bos_idx: int = self.vocab[bos_token] if bos_token and self.vocab[bos_token] != self.unk_idx else -1
        self.eos_idx: int = self.vocab[eos_token] if eos_token and self.vocab[eos_token] != self.unk_idx else -1
        self.mask_idx: int = self.vocab[mask_token] if mask_token and self.vocab[mask_token] != self.unk_idx else -1

    @torch.jit.export
    def lookup_indices_1d(self, values: List[str]) -> List[int]:
        lookup_indices = self.lookup_indices(values)
        # print(type(lookup_indices))
        # print(type(lookup_indices[0]))
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


token_to_freq = {'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2}
sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
c = OrderedDict(sorted_by_freq_tuples)

v = ScriptVocab(c)
v = torch.jit.script(v)
print(v.lookup_word(0, possible_unk_token='unk'))
print(v.lookup_indices_1d(['not_present', 'world', 'hello']))
print(v.lookup_indices_2d([['not_present', 'world', 'hello']]))
print(v.lookup_words_1d(torch.tensor([0, 1, 2], dtype=torch.int32), [2]))
print(v.lookup_words_1d_cycle_heuristic(torch.tensor([0, 1, 2, 0], dtype=torch.int32), [2], ['unk_a', 'unk_b']))
print(v.unk_idx, v.pad_idx, v.bos_idx, v.eos_idx, v.mask_idx)
