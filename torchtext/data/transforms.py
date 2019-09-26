import _C


__all__ = [
    "simple_tokenizer"
]


def simple_tokenizer():
    """A simple tokenizer that splits a sentence by spaces.

    Argument:
        line: a line of text to tokenize

    Output:
        output: a generator over the tokens
    """

    def _internal_func(txt_iter):
        for line in txt_iter:
            yield _C.split_tokenizer(line)
    return _internal_func
