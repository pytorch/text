import torch
from torchtext.data.utils import get_tokenizer
from torchtext.experimental import functional as F


class TokenizerTransform(torch.nn.Module):
    def __init__(self, tokenizer=get_tokenizer('basic_english')):
        """Initiate Tokenizer transform.

        Arguments:
            tokenizer: a callable object to convert a text string
                to a list of token. Default: 'basic_english' tokenizer
        """

        super(TokenizerTransform, self).__init__()
        self.tokenizer = tokenizer

    def forward(self, str_input):
        """
        Inputs:
            str_input: a text string

        Outputs:
            A list of tokens

        Examples:
            >>> tok_transform = torchtext.experimental.transforms.TokenizerTransform()
            >>> tok_transform('here we are')
            >>> ['here', 'we', 'are']
        """
        # type: (str) -> List[str]
        return self.tokenizer(str_input)


class VocabTransform(torch.nn.Module):
    def __init__(self, vocab):
        """Initiate vocab transform.

        Arguments:
            vocab: a callable object to convert a token to integer.
        """

        super(VocabTransform, self).__init__()
        self.vocab = vocab

    def forward(self, tok_iter):
        """
        Inputs:
            tok_iter: a iterable object for tokens

        Outputs:
            A list of integers

        Examples:
            >>> vocab = {'here': 1, 'we': 2, 'are': 3}
            >>> vocab_transform = torchtext.experimental.transforms.VocabTransform(vocab)
            >>> vocab_transform(['here', 'we', 'are'])
            >>> [1, 2, 3]
        """
        # type: (List[str]) -> List[int]
        return [F.vocab_transform(self.vocab, tok) for tok in tok_iter]


class ToTensor(torch.nn.Module):
    def __init__(self, dtype=torch.long):
        """Initiate Tensor transform.

        Arguments:
            dtype: the type of output tensor. Default: `torch.long`
        """

        super(ToTensor, self).__init__()
        self.dtype = dtype

    def forward(self, ids_list):
        """
        Inputs:
            ids_list: a list of numbers.

        Outputs:
            A torch.tensor

        Examples:
            >>> totensor = torchtext.experimental.transforms.ToTensor()
            >>> totensor([1, 2, 3])
            >>> tensor([1, 2, 3])
        """
        return torch.tensor(ids_list).to(self.dtype)


class TextSequential(torch.nn.Sequential):
    def __init__(self, *inps):
        """Initiate Sequential modules transform.

        Arguments:
            Modules: nn.Module or transforms
        """

        super(TextSequential, self).__init__(*inps)

    def forward(self, txt_input):
        """
        Inputs:
            input: a text string

        Outputs:
            output defined by the last transform

        Examples:
            >>> from torchtext.experimental.transforms import TokenizerTransform, \
                    VocabTransform, ToTensor, TextSequential
            >>> vocab = {'here': 1, 'we': 2, 'are': 3}
            >>> vocab_transform = VocabTransform(vocab)
            >>> text_transform = TextSequential(TokenizerTransform(),
                                                VocabTransform(vocab),
                                                ToTensor())
            >>> text_transform('here we are')
            >>> tensor([1, 2, 3])
        """
        # type: (str)
        for module in self:
            txt_input = module(txt_input)
        return txt_input


class NGrams(torch.nn.Module):
    def __init__(self, ngrams):
        """Initiate ngram transform.

        Arguments:
            ngrams: the number of ngrams.
        """

        super(NGrams, self).__init__()
        self.ngrams = ngrams

    def forward(self, token_list):
        """
        Inputs:
            token_list: A list of tokens

        Outputs:
            A list of ngram strings

        Examples:
            >>> token_list = ['here', 'we', 'are']
            >>> ngram_transform = torchtext.experimental.transforms.NGrams(3)
            >>> ngram_transform(token_list)
            >>> ['here', 'we', 'are', 'here we', 'we are', 'here we are']
        """
        _token_list = []
        for _i in range(self.ngrams + 1):
            _token_list += zip(*[token_list[i:] for i in range(_i)])
        return [' '.join(x) for x in _token_list]
