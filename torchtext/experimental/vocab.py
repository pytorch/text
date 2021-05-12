import logging
from collections import Counter, OrderedDict
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
    _load_vocab_from_file,
    _build_vocab_from_text_file
)

__all__ = [
    'build_vocab_from_text_file',
    'load_vocab_from_file',
    'vocab',
]
logger = logging.getLogger(__name__)


def build_vocab_from_text_file(file_path, jited_tokenizer, min_freq=1, unk_token='<unk>', num_cpus=4):
    r"""Create a `Vocab` object from a raw text file.

    The `file_path` can contain any raw text. This function applies a generic JITed tokenizer in
    parallel to the text.

    Args:
        file_object (FileObject): a file object to read data from.
        jited_tokenizer (ScriptModule): a tokenizer that has been JITed using `torch.jit.script`
        min_freq: The minimum frequency needed to include a token in the vocabulary.
            Values less than 1 will be set to 1. Default: 1.
        unk_token: The default unknown token to use. Default: '<unk>'. If not found in text file, it will be inserted to index 0.
        num_cpus (int): the number of cpus to use when loading the vectors from file. Default: 4.

    Returns:
        torchtext.experimental.vocab.Vocab: a `Vocab` object.

    Examples:
        >>> from torchtext.experimental.vocab import build_vocab_from_text_file
        >>> from torchtext.experimental.transforms import basic_english_normalize
        >>> tokenizer = basic_english_normalize()
        >>> tokenizer = basic_english_normalize()
        >>> jit_tokenizer = torch.jit.script(tokenizer)
        >>> v = build_vocab_from_text_file('vocab.txt', jit_tokenizer)
    """
    vocab_obj = _build_vocab_from_text_file(file_path, min_freq, num_cpus, jited_tokenizer)
    if unk_token not in vocab_obj:
        vocab_obj.insert_token(unk_token, 0)
    return Vocab(vocab_obj)


def load_vocab_from_file(file_path, min_freq=1, unk_token='<unk>', num_cpus=4):
    r"""Create a `Vocab` object from a text file.
    The `file_path` should contain tokens separated by new lines.
    Format for txt file:

        token1
        token2
        ...
        token_n

    Args:
        file_object (FileObject): a file like object to read data from.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
            Values less than 1 will be set to 1. Default: 1.
        unk_token: The default unknown token to use. Default: '<unk>'. If not found in vocab file, it will be inserted to index 0.
        num_cpus (int): the number of cpus to use when loading the vectors from file. Default: 4.

    Returns:
        torchtext.experimental.vocab.Vocab: a `Vocab` object.

    Examples:
        >>> from torchtext.experimental.vocab import load_vocab_from_file
        >>> v = load_vocab_from_file('vocab.txt')
    """

    vocab_obj = _load_vocab_from_file(file_path, min_freq, num_cpus)
    if unk_token not in vocab_obj:
        vocab_obj.insert_token(unk_token, 0)
    return Vocab(vocab_obj)


def build_vocab_from_iterator(iterator, min_freq=1, unk_token='<unk>'):
    """
    Build a Vocab from an iterator.

    Args:
        iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
            Values less than 1 will be set to 1. Default: 1.
        unk_token: The default unknown token to use. Default: '<unk>'.
    """

    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    word_vocab = vocab(ordered_dict, min_freq=min_freq, unk_token=unk_token)
    return word_vocab


def vocab(ordered_dict, min_freq=1, unk_token='<unk>'):
    r"""Factory method for creating a vocab object which maps tokens to indices.

    Note that the ordering in which key value pairs were inserted in the `ordered_dict` will be respected when building the vocab.
    Therefore if sorting by token frequency is important to the user, the `ordered_dict` should be created in a way to reflect this.
    Additionally, the if the `unk_token` isn't found inside of the `ordered_dict`, it will be added to the end of the vocab.

    Args:
        ordered_dict (collections.OrderedDict): object holding the frequencies of each token found in the data.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
            Values less than 1 will be set to 1. Default: 1.
        unk_token: The default unknown token to use. Default: '<unk>'. If not found in ordered_dict, it will be inserted at index 0.

    Raises:
        ValueError: if a default `unk_token` isn't provided.

    Examples:
        >>> from torchtext.experimental.vocab import vocab
        >>> from collections import Counter, OrderedDict
        >>> counter = Counter(["a", "a", "b", "b", "b"])
        >>> sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        >>> ordered_dict = OrderedDict(sorted_by_freq_tuples)
        >>> v1 = vocab(ordered_dict)
        >>> tokens = ['e', 'd', 'c', 'b', 'a']
        >>> v2 = vocab(OrderedDict([(token, 1) for token in tokens]))
    """

    if not unk_token:
        raise ValueError("A default unk token wasn't provided.")

    tokens = []
    for token, freq in ordered_dict.items():
        if freq >= min_freq:
            tokens.append(token)

    if unk_token not in tokens:
        tokens.insert(0, unk_token)
    return Vocab(VocabPybind(tokens, None))
