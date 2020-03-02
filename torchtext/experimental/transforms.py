import torch
from torchtext.data.utils import get_tokenizer


class TokenizerTransform(object):

    def __init__(self, tokenizer_name):
        self.tokenizer = get_tokenizer(tokenizer_name)

    def __call__(self, str_input):
        return self.tokenizer(str_input)


class VocabTransform(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, tok_iter):
        return [self.vocab[tok] for tok in tok_iter]


class ToTensor(object):
    def __init__(self, dtype=torch.long):
        self.dtype = dtype

    def __call__(self, ids_list):
        return torch.tensor(ids_list).to(self.dtype)


# Fork from torchvision
class Compose(object):
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
        self.transforms = transforms

    def __call__(self, img):
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


class ListTransform(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, corpus_iter):
        return [self.transforms(item) for item in corpus_iter]
