import random
import _C
from contextlib import contextmanager
from copy import deepcopy
import re

from functools import partial


def _split_tokenizer(x):
    return x.split()


def _spacy_tokenize(x, spacy):
    return [tok.text for tok in spacy.tokenizer(x)]


_patterns = [r'\'',
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

_replacements = [' \'  ',
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

_patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))


def get_tokenizer(tokenizer, language='en'):
    r"""
    Generate tokenizer function for a string sentence.

    Arguments:
        tokenizer: the name of tokenizer function. If None, it returns split()
            function, which splits the string sentence by space.
            If basic_english, it returns _basic_english_normalize() function,
            which normalize the string first and split by space. If a callable
            function, it will return the function. If a tokenizer library
            (e.g. spacy, moses, toktok, revtok, subword), it returns the
            corresponding library.
        language: Default en

    Examples:
        >>> import torchtext
        >>> from torchtext.data import get_tokenizer
        >>> tokenizer = get_tokenizer("basic_english")
        >>> tokens = tokenizer("You can now install TorchText using pip!")
        >>> tokens
        >>> ['you', 'can', 'now', 'install', 'torchtext', 'using', 'pip', '!']

    """

    # default tokenizer is string.split(), added as a module function for serialization
    if tokenizer is None:
        return _split_tokenizer

    if tokenizer == "basic_english":
        if language != 'en':
            raise ValueError("Basic normalization is only available for Enlish(en)")
        # import os
        # Exception: ('emm', [<module '_C'
        # (/home/travis/.cache/Python-Eggs/torchtext-0.4.0-py3.5-linux-x86_64.egg-tmp/
        # _C.cpython-35m-x86_64-linux-gnu.so)>,
        # '/home/travis/.cache/Python-Eggs/torchtext-0.4.0-py3.5-linux-x86_64.egg-tmp/
        # _C.cpython-35m-x86_64-linux-gnu.so',
        # ['__doc__', '__package__', '__spec__', 'basic_english_normalize',
        # '__name__', '__file__', '__loader__'],
        # <built-in method basic_english_normalize of PyCapsule object at
        # 0x7f8e14626c60>])
        # raise Exception('emm', [_C, _C.__file__, _C.__dir__(),
        # _C.basic_english_normalize])
        # ff = _C.__file__
        # f = open(f, "r")
        # s = os.path.exists(_C.__file__)
        # Exception: (True, '/home/travis/.cache/Python-Eggs/
        # torchtext-0.4.0-py3.5-linux-x86_64.egg-tmp',
        # ['_C.cpython-35m-x86_64-linux-gnu.so'])
        # raise Exception(s, os.path.dirname(ff), os.listdir(os.path.dirname(ff)))
        # return _C.basic_english_normalize
        return _C.x

    # simply return if a function is passed
    if callable(tokenizer):
        return tokenizer

    if tokenizer == "spacy":
        try:
            import spacy
            spacy = spacy.load(language)
            return partial(_spacy_tokenize, spacy=spacy)
        except ImportError:
            print("Please install SpaCy. "
                  "See the docs at https://spacy.io for more information.")
            raise
        except AttributeError:
            print("Please install SpaCy and the SpaCy {} tokenizer. "
                  "See the docs at https://spacy.io for more "
                  "information.".format(language))
            raise
    elif tokenizer == "moses":
        try:
            from sacremoses import MosesTokenizer
            moses_tokenizer = MosesTokenizer()
            return moses_tokenizer.tokenize
        except ImportError:
            print("Please install SacreMoses. "
                  "See the docs at https://github.com/alvations/sacremoses "
                  "for more information.")
            raise
    elif tokenizer == "toktok":
        try:
            from nltk.tokenize.toktok import ToktokTokenizer
            toktok = ToktokTokenizer()
            return toktok.tokenize
        except ImportError:
            print("Please install NLTK. "
                  "See the docs at https://nltk.org  for more information.")
            raise
    elif tokenizer == 'revtok':
        try:
            import revtok
            return revtok.tokenize
        except ImportError:
            print("Please install revtok.")
            raise
    elif tokenizer == 'subword':
        try:
            import revtok
            return partial(revtok.tokenize, decap=True)
        except ImportError:
            print("Please install revtok.")
            raise
    raise ValueError("Requested tokenizer {}, valid choices are a "
                     "callable that takes a single string as input, "
                     "\"revtok\" for the revtok reversible tokenizer, "
                     "\"subword\" for the revtok caps-aware tokenizer, "
                     "\"spacy\" for the SpaCy English tokenizer, or "
                     "\"moses\" for the NLTK port of the Moses tokenization "
                     "script.".format(tokenizer))


def is_tokenizer_serializable(tokenizer, language):
    """Extend with other tokenizers which are found to not be serializable
    """
    if tokenizer == 'spacy':
        return False
    return True


def interleave_keys(a, b):
    """Interleave bits from two sort keys to form a joint sort key.

    Examples that are similar in both of the provided keys will have similar
    values for the key defined by this function. Useful for tasks with two
    text fields like machine translation or natural language inference.
    """
    def interleave(args):
        return ''.join([x for t in zip(*args) for x in t])
    return int(''.join(interleave(format(x, '016b') for x in (a, b))), base=2)


def get_torch_version():
    import torch
    v = torch.__version__
    version_substrings = v.split('.')
    major, minor = version_substrings[0], version_substrings[1]
    return int(major), int(minor)


def dtype_to_attr(dtype):
    # convert torch.dtype to dtype string id
    # e.g. torch.int32 -> "int32"
    # used for serialization
    _, dtype = str(dtype).split('.')
    return dtype


# TODO: Write more tests!
def ngrams_iterator(token_list, ngrams):
    """Return an iterator that yields the given tokens and their ngrams.

    Arguments:
        token_list: A list of tokens
        ngrams: the number of ngrams.

    Examples:
        >>> token_list = ['here', 'we', 'are']
        >>> list(ngrams_iterator(token_list, 2))
        >>> ['here', 'here we', 'we', 'we are', 'are']
    """

    def _get_ngrams(n):
        return zip(*[token_list[i:] for i in range(n)])

    for x in token_list:
        yield x
    for n in range(2, ngrams + 1):
        for x in _get_ngrams(n):
            yield ' '.join(x)


class RandomShuffler(object):
    """Use random functions while keeping track of the random state to make it
    reproducible and deterministic."""

    def __init__(self, random_state=None):
        self._random_state = random_state
        if self._random_state is None:
            self._random_state = random.getstate()

    @contextmanager
    def use_internal_state(self):
        """Use a specific RNG state."""
        old_state = random.getstate()
        random.setstate(self._random_state)
        yield
        self._random_state = random.getstate()
        random.setstate(old_state)

    @property
    def random_state(self):
        return deepcopy(self._random_state)

    @random_state.setter
    def random_state(self, s):
        self._random_state = s

    def __call__(self, data):
        """Shuffle and return a new list."""
        with self.use_internal_state():
            return random.sample(data, len(data))
