# coding: utf8
import torch
import warnings


class RawField(object):
    def __init__(self, preprocessing=None, postprocessing=None, is_target=False):
        warnings.warn('{} class has '.format(self.__class__.__name__) +
                      'been retired and moved to torchtext.legacy. Please ' +
                      'import from torchtext.legacy.data if you still want it.', UserWarning)
        raise ImportWarning


class Field(RawField):
    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False,
                 tokenize=None, tokenizer_language='en', include_lengths=False,
                 batch_first=False, pad_token="<pad>", unk_token="<unk>",
                 pad_first=False, truncate_first=False, stop_words=None,
                 is_target=False):
        warnings.warn('{} class has '.format(self.__class__.__name__) +
                      'been retired and moved to torchtext.legacy. Please ' +
                      'import from torchtext.legacy.data if you still want it.', UserWarning)
        raise ImportWarning


class ReversibleField(Field):
    def __init__(self, **kwargs):
        warnings.warn('{} class has '.format(self.__class__.__name__) +
                      'been retired and moved to torchtext.legacy. Please ' +
                      'import from torchtext.legacy.data if you still want it.', UserWarning)
        raise ImportWarning


class SubwordField(ReversibleField):
    def __init__(self, **kwargs):
        warnings.warn('{} class has '.format(self.__class__.__name__) +
                      'been retired and moved to torchtext.legacy. Please ' +
                      'import from torchtext.legacy.data if you still want it.', UserWarning)
        raise ImportWarning


class NestedField(Field):
    def __init__(self, nesting_field, use_vocab=True, init_token=None, eos_token=None,
                 fix_length=None, dtype=torch.long, preprocessing=None,
                 postprocessing=None, tokenize=None, tokenizer_language='en',
                 include_lengths=False, pad_token='<pad>',
                 pad_first=False, truncate_first=False):
        warnings.warn('{} class has '.format(self.__class__.__name__) +
                      'been retired and moved to torchtext.legacy. Please ' +
                      'import from torchtext.legacy.data if you still want it.', UserWarning)
        raise ImportWarning


class LabelField(Field):
    def __init__(self, **kwargs):
        warnings.warn('{} class has '.format(self.__class__.__name__) +
                      'been retired and moved to torchtext.legacy. Please ' +
                      'import from torchtext.legacy.data if you still want it.', UserWarning)
        raise ImportWarning
