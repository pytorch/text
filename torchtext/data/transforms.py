import sentencepiece as spm


__all__ = [
    "sentencepiece_encode_as_ids", "sentencepiece_encode_as_pieces"
]


def sentencepiece_encode_as_ids(spm_path):
    """A sentencepiece tokenizer to convert a text sentence into
       a generator over the ids.

    Arguments:
        sp_path: the file path saving the sentencepiece model.
        txt_iter: input sentence text generator.

    Outputs:
        output: a generator over the ids based on SentencePiece model.

    Examples:
        >>> from torchtext.data.transforms import sentencepiece_encode_as_ids
        >>> sp_id_generator = sentencepiece_encode_as_ids("m_user.model")
        >>> list_a = ["sentencepiece encode as pieces", "examples to   try!"]
        >>> list(sp_id_generator(list_a))
            [[9858, 9249, 1629, 1305, 1809, 53, 842],
             [2347, 13, 9, 150, 37]]

    """

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(spm_path)

    def _internal_func(txt_iter):
        for line in txt_iter:
            yield sp_model.EncodeAsIds(line)
    return _internal_func


def sentencepiece_encode_as_pieces(spm_path):
    """A sentencepiece model for tokenizing a text sentence into
       a generator over the tokens.

    Arguments:
        sp_path: the file path saving the sentencepiece model.
        txt_iter: input sentence text generator.

    Outputs:
        output: a generator over the ids based on SentencePiece model.

    Examples:
        >>> from torchtext.data.transforms import sentencepiece_encode_as_pieces
        >>> sp_tokens_generator = sentencepiece_encode_as_pieces("m_user.model")
        >>> list_a = ["sentencepiece encode as pieces", "examples to   try!"]
        >>> list(sp_tokens_generator(list_a))
            [['▁sentence', 'piece', '▁en', 'co', 'de', '▁as', '▁pieces'],
             ['▁example', 's', '▁to', '▁try', '!']]

    """

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(spm_path)

    def _internal_func(txt_iter):
        for line in txt_iter:
            yield sp_model.EncodeAsPieces(line)
    return _internal_func
