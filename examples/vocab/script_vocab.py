from collections import OrderedDict
import torch
from torchtext.experimental.vocab import Vocab
from typing import List, Optional


class ScriptVocab(Vocab):
    def __init__(self, ordered_dict, **kwargs):
        super(ScriptVocab, self).__init__(ordered_dict, **kwargs)
        self.unk_token: str = kwargs.get('unk_token', '<unk>')
        self.unk_idx: int = self.vocab[self.unk_token]

    @torch.jit.export
    def lookup_indices_1d(self, values: List[str]) -> List[int]:
        return super().lookup_indices(values)

    @torch.jit.export
    def lookup_indices_2d(self, values: List[List[str]]) -> List[List[int]]:
        result = []
        for value in values:
            result.append(super().lookup_indices(value))
        return result

    @torch.jit.export
    def lookup_word(self, idx: int, possible_unk_token: Optional[str] = None) -> str:
        # print(idx, possible_unk_token)
        word = super().lookup_token(idx)
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
        result = []
        filter_token_set = set(filter_token_list)

        for idx in range(values.size(0)):
            value = int(values[idx])
            if value not in filter_token_set:
                token = super().lookup_token(value)
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
        filter_token_set = set(filter_token_list)

        result = []
        for idx in range(values.size(0)):
            value = int(values[idx])
            if value not in filter_token_set:
                if value < vocab_length and value != self.unk_idx:
                    result.append(super().lookup_token(value))
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

v = ScriptVocab(c, unk_token="__UNK__")
# v = torch.jit.script(v)
print(v.lookup_word(0, possible_unk_token='unk'))
print(v.lookup_indices_1d(['not_present', 'world', 'hello']))
print(v.lookup_indices_2d([['not_present', 'world', 'hello']]))
print(v.lookup_words_1d(torch.tensor([0, 1, 2], dtype=torch.int32), [2]))
print(v.lookup_words_1d_cycle_heuristic(torch.tensor([0, 1, 2, 0], dtype=torch.int32), [2], ['unk_a', 'unk_b']))

# v = ScriptVocab(c)
# jit_v = torch.jit.script(v)
# print(v.lookup_word('test'))
