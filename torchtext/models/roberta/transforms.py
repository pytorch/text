import os
import torch
from torch.nn import Module
from torchtext._download_hooks import load_state_dict_from_url
from torchtext import transforms
from torchtext import functional

from typing import List, Union


class XLMRobertaModelTransform(Module):
    def __init__(
        self,
        vocab_path: str,
        spm_model_path: str,
        bos_token: str = "<s>",
        cls_token: str = "<s>",
        pad_token: str = "<pad>",
        eos_token: str = "</s>",
        sep_token: str = "</s>",
        unk_token: str = "<unk>",
        mask_token: str = "<mask>",
        max_seq_len: int = 514,
    ):
        super().__init__()
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.max_seq_len = max_seq_len

        self.tokenizer = transforms.SentencePieceTokenizer(spm_model_path)

        if os.path.exists(vocab_path):
            self.vocab = torch.load(vocab_path)
        else:
            self.vocab = load_state_dict_from_url(vocab_path)

        self.vocab_transform = transforms.VocabTransform(self.vocab)
        self.pad_idx = self.vocab[self.pad_token]
        self.bos_idx = self.vocab[self.bos_token]
        self.eos_idx = self.vocab[self.eos_token]

    def forward(self, input: Union[str, List[str]],
                add_bos: bool = True,
                add_eos: bool = True,
                truncate: bool = True) -> Union[List[int], List[List[int]]]:

        tokens = self.tokenizer(input)

        if truncate:
            tokens = functional.truncate(tokens, self.max_seq_len - 2)

        tokens = self.vocab_transform((tokens))

        if add_bos:
            tokens = functional.add_token(tokens, self.bos_idx)

        if add_eos:
            tokens = functional.add_token(tokens, self.eos_idx, begin=False)

        return tokens


def get_xlmr_transform(vocab_path, spm_model_path, **kwargs) -> XLMRobertaModelTransform:
    return XLMRobertaModelTransform(vocab_path, spm_model_path, **kwargs)
