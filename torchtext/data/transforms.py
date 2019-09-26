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

    Examples:
        >>>import torchtext
        >>>from torchtext.data.transforms import simple_tokenizer
        >>>docs = ["You can now install TorchText using pip!"]
        >>>tokenizer = simple_tokenizer()
        >>>iter = tokenizer(docs)
        >>>next(iter)
           ['You', 'can', 'now', 'install', 'TorchText', 'using', 'pip!']
    """

    def _internal_func(txt_iter):
        for line in txt_iter:
            yield _C.split_tokenizer(line)
    return _internal_func
