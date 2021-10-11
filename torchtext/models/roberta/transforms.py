from typing import List, Any
from torch.nn import Module
from torch.hub import load_state_dict_from_url


class XLMRobertaModelTransform(Module):
    padding_idx: int
    bos_idx: int
    eos_idx: int
    _vocab: Module
    _sp_model: Any

    def __init__(
        self,
        spm_model_url: str,
        vocab_url: str,
        *,
        bos_token: str = "<s>",
        cls_token: str = "<s>",
        pad_token: str = "<pad>",
        eos_token: str = "</s>",
        sep_token: str = "</s>",
        unk_token: str = "<unk>",
        mask_token: str = "<mask>",
        max_seq_len: int = 514,
        truncate: bool = True,
    ):
        super().__init__()
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.truncate = truncate
        self.max_seq_len = max_seq_len
        self.spm_model_url = spm_model_url
        self.vocab_url = vocab_url

    def forward(self, input: str) -> List[int]:
        tokens: List[int] = [self.bos_idx] + self.vocab(self.sp_model.EncodeAsPieces(input))
        if self.truncate:
            tokens = tokens[: self.max_seq_len - 2]
        tokens.append(self.eos_idx)
        return tokens

    def load_state(self):
        self.sp_model = load_state_dict_from_url(self.spm_model_url)
        self.vocab = load_state_dict_from_url(self.vocab_url)
        self.padding_idx = self.vocab[self.pad_token]
        self.bos_idx = self.vocab[self.bos_token]
        self.eos_idx = self.vocab[self.eos_token]


def get_xlmr_transform(spm_model_url, vocab_url, *, **kwargs) -> XLMRobertaModelTransform:
    transform = XLMRobertaModelTransform(spm_model_url, vocab_url, **kwargs)
    transform.load_state()
    return transform
