import torch


def totensor(dtype):
    def _forward(ids_list):
        return torch.tensor(ids_list).to(dtype)

    return _forward


def ngrams_func(ngrams):
    def _forward(token_list):
        _token_list = []
        for _i in range(ngrams + 1):
            _token_list += zip(*[token_list[i:] for i in range(_i)])
        return [" ".join(x) for x in _token_list]

    return _forward


def sequential_transforms(*transforms):
    def _forward(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return _forward
