import torch
from typing import List, Tuple


def regex_tokenizer(line: str, patterns_list: List[Tuple[str, str]]) -> List[str]:
    r"""Regex tokenizer for a string sentence.

    Steps include
    - lowercasing
    - applying all regex replacements defines in patterns_list

    Args:
        line (str): a line of text to tokenize.
        patterns_list (List[Tuple[str, str]]): a list of tuples (ordered pairs) which contain the regex pattern string as the first element and the replacement string as the second element.

    Returns:
        tokens (List[str]): a list of tokens after splitting on whitespace.
    """

    line = line.lower()
    regex = torch.classes.torchtext.Regex('')
    for pattern_re, replaced_str in patterns_list:
        regex.UpdateRe(pattern_re)
        line = regex.Sub(line, replaced_str)
    return line.split()


def vocab_func(vocab):
    def _forward(tok_iter):
        return [vocab[tok] for tok in tok_iter]

    return _forward


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
