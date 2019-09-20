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
