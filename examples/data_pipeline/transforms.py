import torch.nn as nn
from torchtext.data.functional import load_sp_model
from torchtext.experimental.vocab import Vocab
from typing import List
from collections import OrderedDict
import torch
from torch import Tensor
from typing import Any, overload, TypeVar, Union, Iterator
from torch._jit_internal import _copy_to_script_wrapper

T = TypeVar('T')


class SequentialPipeline(nn.Module):
    r"""Text classification pipeline template
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def forward(self, line: str) -> List[int]:
        for _transform in self.transforms:
            line = _transform(line)
        return line


class TextClassificationPipeline(nn.Module):
    r"""Text classification pipeline template
    """

    def __init__(self, label_transform, text_transform):
        super(TextClassificationPipeline, self).__init__()
        self.label_transform = label_transform
        self.text_transform = text_transform

    def forward(self, label_text_tuple):
        return self.label_transform(label_text_tuple[0]), self.text_transform(label_text_tuple[1])


class TextDataPipeline(nn.Module):
    r"""Text data pipeline template
    """

    def __init__(self, tokenizer, vocab):
        super(TextDataPipeline, self).__init__()
        self.tokenizer = tokenizer
        self.vocab = vocab

    @torch.jit.export
    def forward(self, line: str):
        tokens = self.tokenizer(line)
        index = self.vocab(tokens)
        return index


class PretrainedSPTokenizer(nn.Module):
    r"""Tokenizer based on a pretained sentencepiece model
    """

    def __init__(self, spm_file):
        super(PretrainedSPTokenizer, self).__init__()
        self.sp_model = load_sp_model(spm_file)

    def forward(self, line: str) -> List[str]:
        r"""

        """

        return self.sp_model.EncodeAsPieces(line)


class PretrainedSPVocab(nn.Module):
    r"""Vocab based on a pretained sentencepiece model
    """

    def __init__(self, spm_file):
        super(PretrainedSPVocab, self).__init__()
        self.sp_model = load_sp_model(spm_file)
        unk_id = self.sp_model.unk_id()
        unk_token = self.sp_model.IdToPiece(unk_id)
        vocab_list = [self.sp_model.IdToPiece(i) for i in range(self.sp_model.GetPieceSize())]
        self.vocab = Vocab(OrderedDict([(token, 1) for token in vocab_list]), unk_token=unk_token)

    def forward(self, tokens: List[str]) -> List[int]:
        return self.vocab.lookup_indices(tokens)

    def insert_token(self, token: str, index: int) -> None:
        self.vocab.insert_token(token, index)


class VocabTransform(nn.Module):
    r"""Vocab transform
    """

    def __init__(self, vocab):
        super(VocabTransform, self).__init__()
        self.vocab = vocab

    def forward(self, tokens: List[str]) -> List[int]:
        return self.vocab.lookup_indices(tokens)


class PyTextVocabTransform(nn.Module):
    r"""Vocab transform
    """

    def __init__(self, vocab):
        super(PyTextVocabTransform, self).__init__()
        self.vocab = vocab

    def forward(self, tokens: List[str]) -> List[int]:
        return self.vocab.lookup_indices_1d(tokens)


class VectorTransform(nn.Module):
    r"""Vector transform
    """

    def __init__(self, vector):
        super(VectorTransform, self).__init__()
        self.vector = vector

    def forward(self, tokens: List[str]) -> Tensor:
        return self.vector.lookup_vectors(tokens)


class Sequential(nn.Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    To make it easier to understand, here is a small example::
        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )
        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    @overload
    def __init__(self, *args: nn.Module) -> None:
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]') -> None:
        ...

    def __init__(self, *args: Any):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self: T, idx) -> T:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: nn.Module) -> None:
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self._modules.values())

    # NB: We can't really type check this function as the type of input
    # may change dynamically (as is tested in
    # TestScript.test_sequential_intermediary_types).  Cannot annotate
    # with Any as TorchScript expects a more precise type
    def forward(self, input: str):
        for module in self:
            input = module(input)
        return input
