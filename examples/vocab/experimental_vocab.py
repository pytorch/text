from collections import OrderedDict
from fairseq.data.dictionary import Dictionary
from torchtext.experimental.vocab import Vocab


def build_fairseq_vocab(
    vocab_file: str,
    dictionary_class: Dictionary = Dictionary,
    special_token_replacements: Dict[str, SpecialToken] = None,
    unk_token: str = "<unk>",
    max_vocab: int = -1,
    min_count: int = -1,
    tokens_to_add: Optional[List[str]] = None,
):
    """
    Function builds a torchtext Vocab for models pre-trained using Fairseq
    modules. The dictionary class can take any Fairseq Dictionary class
    and is used to load the vocab file.
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
        specials = tuple(special_pair[1] for special_pair in special_token_replacements if special_pair[0] != unk_token)
        # special_tokens_to_remove = ('<pad>', "<s>", "</s>", "<unk>", "<mask>")
        # specials = ('__PAD__', '__BEGIN_OF_SENTENCE__', '__END_OF_SENTENCE__', '__MASK__')

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
        ordered_dict = OrderedDict(dictionary_items)

        # remove specials from dict since Vocab expects a seperate tuple of special tokens
        for s in special_tokens_to_remove:
            if s in ordered_dict:
                del ordered_dict[s]

        return Vocab(dictionary_items, unk_token=unk_replacement, specials=specials)


class ScriptVocab(Vocab):
    def __init__(unk_token='<unk>', **kwargs):
        super().__init__(unk_token=unk_token, **kwargs)
        torch.jit.Attribute(self.unk_token) = unk_token

    @torch.jit.script_method
    def lookup_indices_1d(self, values: List[str]) -> List[int]:
        return super().lookupIndices

    @torch.jit.script_method
    def lookup_word(self, idx: int, possible_unk_token: Optional[str] = None):
        word = super().lookupToken(idx)
        if word != self.unk_token or not possible_unk_token:
            return word
        return possible_unk_token
