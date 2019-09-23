__all__ = [
    "sentencepiece_encode_as_ids"
]


def sentencepiece_encode_as_ids(sp_model, txt_str):
    """A sentencepiece tokenizer to convert a text sentence into
       a list of integers.

    Arguments:
        sp_model: the sentencepiece model.
        txt_str: input sentence text.

    Outputs:
        output: a list of integers based on SentencePiece model.
    """

    return sp_model.EncodeAsIds(txt_str)


def sentencepiece_encode_as_pieces(sp_model, txt_str):
    """A sentencepiece model for tokenizing a text sentence into
       a list of tokens.

    Arguments:
        sp_model: the sentencepiece model.
        txt_str: input sentence text.

    Outputs:
        output: a list of tokens.
    """

    return sp_model.EncodeAsPieces(txt_str)


def generate_sp_tokenizer(filename, vocab_size=20000,
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
        >>> from torchtext.data.utils import generate_sp_tokenizer
        >>> generate_sp_tokenizer('test.csv', vocab_size=23456,
        >>>                       model_prefix='spm_user')
        >>> import sentencepiece as spm
        >>> sp_user = spm.SentencePieceProcessor()
        >>> sp_user.load('spm_user.model')
    """

    import sentencepiece as spm
    spm_training_string = "--input={} \
                           --vocab_size={} \
                           --model_prefix={} \
                           --model_type={}".format(filename,
                                                   vocab_size,
                                                   model_prefix,
                                                   model_type)
    spm.SentencePieceTrainer.train(spm_training_string)
    return None
