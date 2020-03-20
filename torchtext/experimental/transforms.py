import torch
from torchtext.data.utils import get_tokenizer
from torchtext.experimental import functional as F


class TokenizerTransform(torch.nn.Module):

    def __init__(self, tokenizer=get_tokenizer('basic_english')):
        super(TokenizerTransform, self).__init__()
        self.tokenizer = tokenizer

    def forward(self, str_input):
        # type: (str) -> List[str]
        return self.tokenizer(str_input)


class VocabTransform(torch.nn.Module):
    def __init__(self, vocab):
        super(VocabTransform, self).__init__()
        self.vocab = vocab

    def forward(self, tok_iter):
        # type: (List[str]) -> List[int]
        return [F.vocab_transform(self.vocab, tok) for tok in tok_iter]


class ToTensor(torch.nn.Module):
    def __init__(self, dtype=torch.long):
        super(ToTensor, self).__init__()
        self.dtype = dtype

    def forward(self, ids_list):
        return torch.tensor(ids_list).to(self.dtype)


class NGrams(torch.nn.Module):
    def __init__(self, ngrams):
        super(NGrams, self).__init__()
        self.ngrams = ngrams

    def forward(self, token_list):
        _token_list = []
        for _i in range(self.ngrams + 1):
            _token_list += zip(*[token_list[i:] for i in range(_i)])
        return [' '.join(x) for x in _token_list]
