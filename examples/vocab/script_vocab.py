from collections import OrderedDict
import torch
from torchtext.experimental.vocab import Vocab
from typing import List, Optional


class ScriptVocab(Vocab):
    def __init__(self, ordered_dict, **kwargs):
        super(ScriptVocab, self).__init__(ordered_dict, **kwargs)
        self.unk_token = kwargs.get('unk_token')

    # @torch.jit.export
    # def lookup_indices_1d(self, tokens: List[str]) -> List[int]:
    #     return super().lookupIndices(tokens)

    @torch.jit.export
    def lookup_word(self, idx: int, possible_unk_token: Optional[str] = None) -> str:
        # print(idx, possible_unk_token)
        word = super().lookupToken(idx)
        print(word, self.unk_token)
        if word != self.unk_token or possible_unk_token is None:
            return word
        return possible_unk_token


token_to_freq = {'hello': 4, 'world': 3, 'ᑌᑎIᑕOᗪᕮ_Tᕮ᙭T': 5, 'freq_too_low': 2}
sorted_by_freq_tuples = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
c = OrderedDict(sorted_by_freq_tuples)

v = ScriptVocab(c, unk_token="__UNK__")
print(v.lookup_word(0, possible_unk_token='unk'))
print(v.lookupIndices(['not_present', 'world', 'hello']))

# v = ScriptVocab(c)
# jit_v = torch.jit.script(v)
# print(v.lookup_word('test'))
