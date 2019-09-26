import sentencepiece as spm


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
    """Train a SentencePiece tokenizer.

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
    """Load a  sentencepiece model for file.

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
    """A sentencepiece model to numericalize a text sentence into
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
    """A sentencepiece model to tokenize a text sentence into
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
    """A transform to convert text string

    Examples:
        >>> from torchtext.data.functional import custom_replace
        >>> custom_replace_transform = custom_replace([(r'S', 's'), (r'\s+', ' ')])
        >>> list_a = ["Sentencepiece encode  aS  pieces", "exampleS to   try!"]
        >>> list(custom_replace_transform(list_a))
            ['sentencepiece encode as pieces', 'examples to try!']
    """

    import re
    _patterns = list((re.compile(p), r)
                     for (p, r) in replace_pattern)

    def _internal_func(txt_iter):
        for line in txt_iter:
            for pattern_re, replaced_str in _patterns:
                line = pattern_re.sub(replaced_str, line)
            yield line
    return _internal_func
