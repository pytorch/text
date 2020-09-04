import torch
from torchtext.data.utils import ngrams_iterator


def vocab_func(vocab):
    def func(tokens_list_iter):
        return [vocab[tok] for tokens_list in tokens_list_iter for tok in tokens_list]

    return func


def vector_func(vector):
    def func(tokens_list_iter):
        return [vector.get_vecs_by_tokens(tokens_list) for tokens_list in tokens_list_iter]

    return func


def tokenizer_func(tokenizer):
    def func(lines):
        return [tokenizer(line) for line in lines]

    return func


def totensor(dtype):
    def func(ids_list):
        return torch.tensor(ids_list).to(dtype)

    return func


def ngrams_func(ngrams):
    def func(token_list):
        return list(ngrams_iterator(token_list, ngrams))

    return func


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func
