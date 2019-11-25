import sentencepiece as spm
import re
from torchtext.utils import unicode_csv_reader
import io

__all__ = [
    "generate_sp_model", "load_sp_model",
    "sentencepiece_numericalizer", "sentencepiece_tokenizer"
]


"""
This file contains experimental functionality.
All of these are experimental, unstable, and subject to change or deletion.
"""


def generate_sp_model(filename, vocab_size=20000,
                      model_type="unigram",
                      model_prefix='m_user'):
    r"""Train a SentencePiece tokenizer.

    Arguments:
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

    spm_training_string = "--input={} \
                           --vocab_size={} \
                           --model_prefix={} \
                           --model_type={}".format(filename,
                                                   vocab_size,
                                                   model_prefix,
                                                   model_type)
    spm.SentencePieceTrainer.train(spm_training_string)
    return None


def load_sp_model(spm_path):
    r"""Load a  sentencepiece model for file.

    Arguments:
        spm_path: the file path saving the sentencepiece model.

    Outputs:
        output: a SentencePiece model.

    Examples:
        >>> from torchtext.data.functional import load_sp_model
        >>> sp_model = load_sp_model("m_user.model")
    """

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(spm_path)
    return sp_model


def sentencepiece_numericalizer(sp_model):
    r"""A sentencepiece model to numericalize a text sentence into
       a generator over the ids.

    Arguments:
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

    Arguments:
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


def read_text_iterator(path, tokenizer):
    r"""Read text from path and yield a list of tokens based on the tokenizer

    Arguments:
        path: the file path.
        tokenizer: the tokenizer used to tokenize string text.

    Examples:
        >>> from torchtext.data.functional import read_text_iterator
        >>> tokenizer = get_tokenizer("basic_english")
        >>> list((read_text_iterator('.data/ptb.train.txt', tokenizer)))
            [['Sentencepiece', 'encode', 'as', 'pieces'], ['example', 'to', 'try!']]
    """

    with io.open(path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            tokens = tokenizer(' '.join(row))
            yield tokens


def create_data_from_iterator(vocab, iterator, removed_tokens=None):
    r"""Yield a list of ids from an token iterator with a vocab.

    Arguments:
        vocab: the vocabulary convert token into id.
        iterator: the iterator yield a list of tokens.
        removed_tokens: removed tokens from output dataset (Default: None)

    Examples:
        >>> from torchtext.data.functional import simple_space_split
        >>> from torchtext.data.functional import create_data_from_iterator
        >>> vocab = {'Sentencepiece' : 0, 'encode' : 1, 'as' : 2, 'pieces' : 3}
        >>> ids_iter = create_data_from_iterator(vocab,
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
            yield iter(vocab[token] for token in
                       filter(lambda x: x not in removed_tokens, tokens))
