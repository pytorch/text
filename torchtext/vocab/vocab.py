from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchtext.utils import _log_class_usage


class Vocab(nn.Module):
    __jit_unused_properties__ = ["is_jitable"]
    r"""Creates a vocab object which maps tokens to indices.

    Args:
        vocab (torch.classes.torchtext.Vocab or torchtext._torchtext.Vocab): a cpp vocab object.
    """

    def __init__(self, vocab) -> None:
        super(Vocab, self).__init__()
        self.vocab = vocab
        _log_class_usage(__class__)

    @property
    def is_jitable(self):
        return isinstance(self.vocab, torch._C.ScriptObject)

    @torch.jit.export
    def forward(self, tokens: List[str]) -> List[int]:
        r"""Calls the `lookup_indices` method

        Args:
            tokens: a list of tokens used to lookup their corresponding `indices`.

        Returns:
            The indices associated with a list of `tokens`.
        """
        return self.vocab.lookup_indices(tokens)

    @torch.jit.export
    def __len__(self) -> int:
        r"""
        Returns:
            The length of the vocab.
        """
        return len(self.vocab)

    @torch.jit.export
    def __contains__(self, token: str) -> bool:
        r"""
        Args:
            token: The token for which to check the membership.

        Returns:
            Whether the token is member of vocab or not.
        """
        return self.vocab.__contains__(token)

    @torch.jit.export
    def __getitem__(self, token: str) -> int:
        r"""
        Args:
            token: The token used to lookup the corresponding index.

        Returns:
            The index corresponding to the associated token.
        """
        return self.vocab[token]

    @torch.jit.export
    def set_default_index(self, index: Optional[int]) -> None:
        r"""
        Args:
            index: Value of default index. This index will be returned when OOV token is queried.
        """
        self.vocab.set_default_index(index)

    @torch.jit.export
    def get_default_index(self) -> Optional[int]:
        r"""
        Returns:
            Value of default index if it is set.
        """
        return self.vocab.get_default_index()

    @torch.jit.export
    def insert_token(self, token: str, index: int) -> None:
        r"""
        Args:
            token: The token used to lookup the corresponding index.
            index: The index corresponding to the associated token.
        Raises:
            RuntimeError: If `index` is not in range [0, Vocab.size()] or if `token` already exists in the vocab.
        """
        self.vocab.insert_token(token, index)

    @torch.jit.export
    def append_token(self, token: str) -> None:
        r"""
        Args:
            token: The token used to lookup the corresponding index.

        Raises:
            RuntimeError: If `token` already exists in the vocab
        """
        self.vocab.append_token(token)

    @torch.jit.export
    def lookup_token(self, index: int) -> str:
        r"""
        Args:
            index: The index corresponding to the associated token.

        Returns:
            token: The token used to lookup the corresponding index.

        Raises:
            RuntimeError: If `index` not in range [0, itos.size()).
        """
        return self.vocab.lookup_token(index)

    @torch.jit.export
    def lookup_tokens(self, indices: List[int]) -> List[str]:
        r"""
        Args:
            indices: The `indices` used to lookup their corresponding`tokens`.

        Returns:
            The `tokens` associated with `indices`.

        Raises:
            RuntimeError: If an index within `indices` is not int range [0, itos.size()).
        """
        return self.vocab.lookup_tokens(indices)

    @torch.jit.export
    def lookup_indices(self, tokens: List[str]) -> List[int]:
        r"""
        Args:
            tokens: the tokens used to lookup their corresponding `indices`.

        Returns:
            The 'indices` associated with `tokens`.
        """
        return self.vocab.lookup_indices(tokens)

    @torch.jit.export
    def get_stoi(self) -> Dict[str, int]:
        r"""
        Returns:
            Dictionary mapping tokens to indices.
        """
        return self.vocab.get_stoi()

    @torch.jit.export
    def get_itos(self) -> List[str]:
        r"""
        Returns:
            List mapping indices to tokens.
        """
        return self.vocab.get_itos()

    def __prepare_scriptable__(self):
        r"""Return a JITable Vocab."""
        if not self.is_jitable:
            cpp_vocab = torch.classes.torchtext.Vocab(self.vocab.itos_, self.vocab.default_index_)
            return Vocab(cpp_vocab)
        return self
