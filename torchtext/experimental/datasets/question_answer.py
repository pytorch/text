import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.experimental.datasets import raw


def vocab_func(vocab):
    def _forward(tok_iter):
        return [vocab[tok] for tok in tok_iter]
    return _forward


def totensor(dtype):
    def _forward(ids_list):
        return torch.tensor(ids_list).to(dtype)
    return _forward


def build_vocab(data, transform):
    tok_list = []
    for processed in data:
        ans_token = []
        for item in processed['answers']:
            ans_token += transform(item)
        tok_list.append(transform(processed['context']) +
                        transform(processed['question']) + ans_token)
    return build_vocab_from_iterator(tok_list)


def squential_transforms(*transforms):
    def _forward(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return _forward


class QuestionAnswerDataset(torch.utils.data.Dataset):
    """Defines an abstract question answer datasets.
       Currently, we only support the following datasets:
             - SQuAD1
             - SQuAD2
    """

    def __init__(self, data, vocab, transforms):
        """Initiate question answer dataset.

        Arguments:
            data: a dictionary of data.
                For example {'context': context_data, 'answers': answers_data,
                             'question': question_data, 'ans_pos': ans_pos_data}
                [(label1, text1), (label2, text2), (label2, text3)]
            vocab: Vocabulary object used for dataset.
            transforms: a dictionary of transforms.
                For example {'context': context_transform, 'answers': answers_transform,
                             'question': question_transform, 'ans_pos': ans_pos_transform}
        """

        super(QuestionAnswerDataset, self).__init__()
        self.data = data
        self.vocab = vocab
        self.transforms = transforms

    def __getitem__(self, i):
        _data = {'context': self.transforms['context'](self.data[i]['context']),
                 'question': self.transforms['question'](self.data[i]['question']),
                 'answers': [], 'ans_pos': []}
        for idx in range(len(self.data[i]['answer_start'])):
            _data['answers'].append(self.transforms['answers'](self.data[i]['answers'][idx]))
            ans_start_idx = self.data[i]['answer_start'][idx]
            if ans_start_idx == -1:  # No answer for this sample
                _data['ans_pos'].append(self.transforms['ans_pos']([-1, -1]))
            else:
                ans_start_token_idx = len(self.transforms['context'](self.data[i]['context'][:ans_start_idx]))
                ans_end_token_idx = ans_start_token_idx + \
                    len(self.transforms['answers'](self.data[i]['answers'][idx])) - 1
                _data['ans_pos'].append(self.transforms['ans_pos']([ans_start_token_idx, ans_end_token_idx]))
        return _data

    def __len__(self):
        return len(self.data)

    def get_vocab(self):
        return self.vocab


def _setup_datasets(dataset_name, root='.data', vocab=None,
                    tokenizer=None, data_select=('train', 'dev')):
    text_transform = []
    if tokenizer is None:
        tokenizer = get_tokenizer('basic_english')
    text_transform = squential_transforms(tokenizer)
    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(('train', 'dev'))):
        raise TypeError('Given data selection {} is not supported!'.format(data_select))
    train, dev = DATASETS[dataset_name](root=root)
    raw_data = {'train': [item for item in train],
                'dev': [item for item in dev]}
    if vocab is None:
        if 'train' not in data_select:
            raise TypeError("Must pass a vocab if train is not selected.")
        vocab = build_vocab(raw_data['train'], text_transform)
    text_transform = squential_transforms(text_transform, vocab_func(vocab), totensor(dtype=torch.long))
    transforms = {'context': text_transform,
                  'question': text_transform,
                  'answers': text_transform,
                  'ans_pos': totensor(dtype=torch.long)}
    return tuple(QuestionAnswerDataset(raw_data[item], vocab, transforms) for item in data_select)


def SQuAD1(*args, **kwargs):
    """ Defines SQuAD1 datasets.

    Create question answer dataset: SQuAD1

    Separately returns the train and dev dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'dev'))
            By default, all the two datasets (train, dev) are generated. Users
            could also choose any one of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import SQuAD1
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, dev = SQuAD1()
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, dev = SQuAD1(tokenizer=tokenizer)
    """

    return _setup_datasets(*(('SQuAD1',) + args), **kwargs)


def SQuAD2(*args, **kwargs):
    """ Defines SQuAD2 datasets.

    Create question answer dataset: SQuAD2

    Separately returns the train and dev dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'dev'))
            By default, all the two datasets (train, dev) are generated. Users
            could also choose any one of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import SQuAD2
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, dev = SQuAD2()
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, dev = SQuAD2(tokenizer=tokenizer)
    """
    return _setup_datasets(*(('SQuAD2',) + args), **kwargs)


DATASETS = {
    'SQuAD1': raw.SQuAD1,
    'SQuAD2': raw.SQuAD2
}
