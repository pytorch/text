import _C


__all__ = [
    "simple_tokenizer"
]


def simple_tokenizer():
    """A simple tokenizer that splits a sentence by spaces.

    Argument:
        txt_iter: input sentence text generator

    Output:
        output: a generator over a list of tokens 
    """

    def _internal_func(txt_iter):
        for line in txt_iter:
            yield _C.split_tokenizer(line)
    return _internal_func
