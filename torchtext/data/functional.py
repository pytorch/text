import re
import io
import torch

__all__ = [
    "generate_sp_model", "load_sp_model",
    "sentencepiece_numericalizer", "sentencepiece_tokenizer",
    "numericalize_tokens_from_iterator",
    "filter_wikipedia_xml",
    "to_map_style_dataset",
]


"""
This file contains experimental functionality.
All of these are experimental, unstable, and subject to change or deletion.
"""


def generate_sp_model(filename, vocab_size=20000,
                      model_type="unigram",
                      model_prefix='m_user'):
    r"""Train a SentencePiece tokenizer.

    Args:
        filename: the data file for training SentencePiece model.
        vocab_size: the size of vocabulary (Default: 20,000).
        model_type: the type of SentencePiece model, including unigram,
            bpe, char, word.
        model_prefix: the prefix of the files saving model and vocab.

    Outputs:
        The model and vocab are saved in two separate files with
            model_prefix.

    Examples:
        >>> from torchtext.data.functional import generate_sp_model
        >>> generate_sp_model('test.csv', vocab_size=23456, model_prefix='spm_user')
    """
    torch.ops.torchtext.generate_sp_model(filename, vocab_size, model_type, model_prefix)


def load_sp_model(spm):
    r"""Load a  sentencepiece model for file.

    Args:
        spm: the file path or a file object saving the sentencepiece model.

    Outputs:
        output: a SentencePiece model.

    Examples:
        >>> from torchtext.data.functional import load_sp_model
        >>> sp_model = load_sp_model("m_user.model")
        >>> sp_model = load_sp_model(open("m_user.model", 'rb'))
    """
    if isinstance(spm, str):
        return torch.ops.torchtext.load_sp_model(spm)
    elif isinstance(spm, io.BufferedReader):
        return torch.ops.torchtext.load_sp_model_string(spm.read())
    else:
        raise TypeError(
            f'Unsupported type for spm argument: {type(spm).__name__}. ' +
            'Supported types are: ' +
            ', '.join([
                'str', 'io.BufferedReader'
            ]))


def sentencepiece_numericalizer(sp_model):
    r"""A sentencepiece model to numericalize a text sentence into
       a generator over the ids.

    Args:
        sp_model: a SentencePiece model.

    Outputs:
        output: a generator with the input of text sentence and the output of the
            corresponding ids based on SentencePiece model.

    Examples:
        >>> from torchtext.data.functional import sentencepiece_numericalizer
        >>> sp_id_generator = sentencepiece_numericalizer(sp_model)
        >>> list_a = ["sentencepiece encode as pieces", "examples to   try!"]
        >>> list(sp_id_generator(list_a))
            [[9858, 9249, 1629, 1305, 1809, 53, 842],
             [2347, 13, 9, 150, 37]]
    """

    def _internal_func(txt_iter):
        for line in txt_iter:
            yield sp_model.EncodeAsIds(line)
    return _internal_func


def sentencepiece_tokenizer(sp_model):
    r"""A sentencepiece model to tokenize a text sentence into
       a generator over the tokens.

    Args:
        sp_model: a SentencePiece model.

    Outputs:
        output: a generator with the input of text sentence and the output of the
            corresponding tokens based on SentencePiece model.

    Examples:
        >>> from torchtext.data.functional import sentencepiece_tokenizer
        >>> sp_tokens_generator = sentencepiece_tokenizer(sp_model)
        >>> list_a = ["sentencepiece encode as pieces", "examples to   try!"]
        >>> list(sp_tokens_generator(list_a))
            [['_sentence', 'piece', '_en', 'co', 'de', '_as', '_pieces'],
             ['_example', 's', '_to', '_try', '!']]
    """

    def _internal_func(txt_iter):
        for line in txt_iter:
            yield sp_model.EncodeAsPieces(line)
    return _internal_func


def custom_replace(replace_pattern):
    r"""A transform to convert text string.

    Examples:
        >>> from torchtext.data.functional import custom_replace
        >>> custom_replace_transform = custom_replace([(r'S', 's'), (r'\s+', ' ')])
        >>> list_a = ["Sentencepiece encode  aS  pieces", "exampleS to   try!"]
        >>> list(custom_replace_transform(list_a))
            ['sentencepiece encode as pieces', 'examples to try!']
    """

    _patterns = list((re.compile(p), r)
                     for (p, r) in replace_pattern)

    def _internal_func(txt_iter):
        for line in txt_iter:
            for pattern_re, replaced_str in _patterns:
                line = pattern_re.sub(replaced_str, line)
            yield line
    return _internal_func


def simple_space_split(iterator):
    r"""A transform to split text string by spaces.

    Examples:
        >>> from torchtext.data.functional import simple_space_split
        >>> list_a = ["Sentencepiece encode as pieces", "example to try!"]
        >>> list(simple_space_split(list_a))
            [['Sentencepiece', 'encode', 'as', 'pieces'], ['example', 'to', 'try!']]
    """

    for line in iterator:
        yield line.split()


def numericalize_tokens_from_iterator(vocab, iterator, removed_tokens=None):
    r"""Yield a list of ids from an token iterator with a vocab.

    Args:
        vocab: the vocabulary convert token into id.
        iterator: the iterator yield a list of tokens.
        removed_tokens: removed tokens from output dataset (Default: None)

    Examples:
        >>> from torchtext.data.functional import simple_space_split
        >>> from torchtext.data.functional import numericalize_tokens_from_iterator
        >>> vocab = {'Sentencepiece' : 0, 'encode' : 1, 'as' : 2, 'pieces' : 3}
        >>> ids_iter = numericalize_tokens_from_iterator(vocab,
        >>>                               simple_space_split(["Sentencepiece as pieces",
        >>>                                                   "as pieces"]))
        >>> for ids in ids_iter:
        >>>     print([num for num in ids])
        >>> [0, 2, 3]
        >>> [2, 3]
    """
    for tokens in iterator:
        if removed_tokens is None:
            yield iter(vocab[token] for token in tokens)
        else:
            yield iter(map(lambda x: vocab[x],
                       filter(lambda x: x not in removed_tokens, tokens)))


_patterns = [(r'<.*>', ''),
             (r'&amp;', '&'),
             (r'&lt;', '<'),
             (r'&gt;', '>'),
             (r'<ref[^<]*<\/ref>', ''),
             (r'<[^>]*>', ''),
             (r'\[http:[^] ]*', '['),
             (r'\|thumb', ''),
             (r'\|left', ''),
             (r'\|right', ''),
             (r'\|\d+px', ''),
             (r'\[\[image:[^\[\]]*\|', ''),
             (r'\[\[category:([^|\]]*)[^]]*\]\]', '[[$1]]'),
             (r'\[\[[a-z\-]*:[^\]]*\]\]', ''),
             (r'\[\[[^\|\]]*\|', '[['),
             (r'\{\{[^\}]*\}\}', ''),
             (r'\{[^\}]*\}', ''),
             (r'\[', ''),
             (r'\]', ''),
             (r'&[^;]*;', ' '),
             (r'A', 'a'), (r'B', 'b'), (r'C', 'c'),
             (r'D', 'd'), (r'E', 'e'), (r'F', 'f'),
             (r'G', 'g'), (r'H', 'h'), (r'I', 'i'),
             (r'J', 'j'), (r'K', 'k'), (r'L', 'l'),
             (r'M', 'm'), (r'N', 'n'), (r'O', 'o'),
             (r'P', 'p'), (r'Q', 'q'), (r'R', 'r'),
             (r'S', 's'), (r'T', 't'), (r'U', 'u'),
             (r'V', 'v'), (r'W', 'w'), (r'X', 'x'),
             (r'Y', 'y'), (r'Z', 'z'),
             (r'0', ' zero '), (r'1', ' one '), (r'2', ' two '),
             (r'3', ' three '), (r'4', ' four '), (r'5', ' five '),
             (r'6', ' six '), (r'7', ' seven '), (r'8', ' eight '),
             (r'9', ' nine '),
             (r'[^a-z\n]+', ' '),
             (r'\n ', ''),
             (r'\s+', ' '),
             (r'\n\s*\n', r'\n')
             ]


def filter_wikipedia_xml(text_iterator):
    r"""Filter wikipedia xml lines according to https://github.com/facebookresearch/fastText/blob/master/wikifil.pl

    args:
        text_iterator: An iterator type object that yields strings. Examples include string list, text io, generators etc.

    Examples:
        >>> from torchtext.data.functional import filter_wikipedia_xml
        >>> from torchtext.datasets import EnWik9
        >>> data_iter = EnWik9(split='train')
        >>> filter_data_iter = filter_wikipedia_xml(data_iter)
        >>> file_name = '.data/EnWik9/enwik9'
        >>> filter_data_iter = filter_wikipedia_xml(open(file_name,'r'))
    """

    try:
        iter(text_iterator)
    except:
        raise TypeError("Input {} must support iterator semantics".format(text_iterator))

    norm_transform = custom_replace(_patterns)
    for line in text_iterator:
        if '#redirect' in line or '#REDIRECT' in line:
            continue
        line = list(norm_transform([line]))[0].strip()
        if line:
            yield line


def to_map_style_dataset(iter_data):
    r"""Convert iterable-style dataset to map-style dataset.

    args:
        iter_data: An iterator type object. Examples include Iterable datasets, string list, text io, generators etc.


    Examples:
        >>> from torchtext.datasets import IMDB
        >>> from torchtext.data import to_map_style_dataset
        >>> train_iter = IMDB(split='train')
        >>> train_dataset = to_map_style_dataset(train_iter)
        >>> file_name = '.data/EnWik9/enwik9'
        >>> data_iter = to_map_style_dataset(open(file_name,'r'))
    """

    # Inner class to convert iterable-style to map-style dataset
    class _MapStyleDataset(torch.utils.data.Dataset):

        def __init__(self, iter_data):
            # TODO Avoid list issue #1296
            self._data = list(iter_data)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

    return _MapStyleDataset(iter_data)
