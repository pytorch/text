import sentencepiece as spm

__all__ = [
    "generate_sp_tokenizer"
]


"""
This file contains experimental functionality.
All of these are experimental, unstable, and subject to change or deletion.
"""


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
        >>> from torchtext.data.functional import generate_sp_tokenizer
        >>> generate_sp_tokenizer('test.csv', vocab_size=23456,
        >>>                       model_prefix='spm_user')
        >>> import sentencepiece as spm
        >>> sp_user = spm.SentencePieceProcessor()
        >>> sp_user.load('spm_user.model')
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
