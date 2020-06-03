import torch
import torch.nn as nn
import torchtext
from typing import List, Tuple


_patterns: List['str'] = [
    r'\'',
    r'\"',
    r'\.',
    r'<br \/>',
    r',',
    r'\(',
    r'\)',
    r'\!',
    r'\?',
    r'\;',
    r'\:',
    r'\s+']

_replacements: List['str'] = [
    ' \'  ',
    '',
    ' . ',
    ' ',
    ' , ',
    ' ( ',
    ' ) ',
    ' ! ',
    ' ? ',
    ' ',
    ' ',
    ' ']

_patterns_list: List[Tuple[str, str]] = list(zip(_patterns, _replacements))


class BasicEnglishNormalize(nn.Module):
    r"""
    Basic normalization for a line of text.
    Normalization includes
    - lowercasing
    - complete some basic text normalization for English words as follows:
        add spaces before and after '\''
        remove '\"',
        add spaces before and after '.'
        replace '<br \/>'with single space
        add spaces before and after ','
        add spaces before and after '('
        add spaces before and after ')'
        add spaces before and after '!'
        add spaces before and after '?'
        replace ';' with single space
        replace ':' with single space
        replace multiple spaces with single space

    Returns a list of tokens after splitting on whitespace.
    """

    def __init__(self):
        super(BasicEnglishNormalize, self).__init__()

        self.regex = torch.classes.torchtext.Regex('')
        self.patterns_list = [
            (r'\'', ' \'  '),
            (r'\"', ''),
            (r'\.', ' . '),
            (r'<br \/>', ' '),
            (r',', ' , '),
            (r'\(', ' ( '),
            (r'\)', ' ) '),
            (r'\!', ' ! '),
            (r'\?', ' ? '),
            (r'\;', ' '),
            (r'\:', ' '),
            (r'\s+', ' ')]

    def forward(self, line: str) -> List[str]:
        return self.basic_english_normalize(line)

    def basic_english_normalize(self, line: str) -> List[str]:
        line = line.lower()
        for pattern_re, replaced_str in self.patterns_list:
            # print("patterns: [{}] [{}]".format(pattern_re, replaced_str))
            # print("line before: [{}]".format(line))
            self.regex.UpdateRe(pattern_re)
            line = self.regex.Sub(line, replaced_str)
            # print("line after: [{}]".format(line))

        tokens = line.split(' ')
        sanitized_tokens: List[str] = []
        # tokens = list(filter(lambda token: token != '', tokens))

        for token in tokens:
            if token != '':
                sanitized_tokens.append(token)

        # print("out tokens", tokens)
        return tokens


def regex_tokenizer(line: str, patterns_list: List[Tuple[str, str]]) -> List[str]:
    r"""
    Regex tokenizer for a string sentence.
    Steps include
    - lowercasing
    - applying all regex replacements defines in patterns_list

    Arguments:
    line: a line of text.
    patterns_list: a list of tuples (ordered pairs) which contain the regex pattern string as the first element and the replacement string as the second element.

    Returns a list of tokens after splitting on whitespace following the regex pattern substitutions.
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
