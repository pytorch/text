import torch
from torchtext.data.utils import get_tokenizer
from torchtext.experimental import functional as F


class TokenizerTransform(torch.nn.Module):

    def __init__(self, tokenizer_name=None):
        super(TokenizerTransform, self).__init__()
        self.tokenizer = get_tokenizer(tokenizer_name)

    def forward(self, str_input):
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


# Fork from torchvision
class Compose(torch.nn.Module):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        super(Compose, self).__init__()
        self.transforms = transforms

    def forward(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ListTransform(torch.nn.Module):
    def __init__(self, transforms):
        super(ListTransform, self).__init__()
        self.transforms = transforms

    def forward(self, corpus_iter):
        return [self.transforms(item) for item in corpus_iter]
