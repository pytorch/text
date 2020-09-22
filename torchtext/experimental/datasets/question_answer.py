import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.experimental.datasets.raw import question_answer as raw
from torchtext.experimental.functional import (
    totensor,
    vocab_func,
    sequential_transforms,
)


class QuestionAnswerDataset(torch.utils.data.Dataset):
    """Defines an abstract question answer datasets.
       Currently, we only support the following datasets:
             - SQuAD1
             - SQuAD2
    """

    def __init__(self, data, vocab, transforms):
        """Initiate question answer dataset.

        Arguments:
            data: a tuple of (context, question, answers, ans_pos).
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
        raw_context, raw_question, raw_answers, raw_answer_start = self.data[i]
        _context = self.transforms['context'](raw_context)
        _question = self.transforms['question'](raw_question)
        _answers, _ans_pos = [], []
        for idx in range(len(raw_answer_start)):
            _answers.append(self.transforms['answers'](raw_answers[idx]))
            ans_start_idx = raw_answer_start[idx]
            if ans_start_idx == -1:  # No answer for this sample
                _ans_pos.append(self.transforms['ans_pos']([-1, -1]))
            else:
                ans_start_token_idx = len(self.transforms['context'](raw_context[:ans_start_idx]))
                ans_end_token_idx = ans_start_token_idx + \
                    len(self.transforms['answers'](raw_answers[idx])) - 1
                _ans_pos.append(self.transforms['ans_pos']([ans_start_token_idx, ans_end_token_idx]))
        return (_context, _question, _answers, _ans_pos)

    def __len__(self):
        return len(self.data)

    def get_vocab(self):
        return self.vocab


def _setup_datasets(dataset_name,
                    root='.data',
                    vocab=None,
                    tokenizer=None,
                    data_select=('train', 'dev')):
    text_transform = []
    if tokenizer is None:
        tokenizer = get_tokenizer('basic_english')
    text_transform = sequential_transforms(tokenizer)
    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(('train', 'dev'))):
        raise TypeError('Given data selection {} is not supported!'.format(data_select))
    train, dev = raw.DATASETS[dataset_name](root=root)
    raw_data = {'train': [item for item in train],
                'dev': [item for item in dev]}
    if vocab is None:
        if 'train' not in data_select:
            raise TypeError("Must pass a vocab if train is not selected.")

        def apply_transform(data):
            for (_context, _question, _answers, _ans_pos) in data:
                tok_ans = []
                for item in _answers:
                    tok_ans += text_transform(item)
                yield text_transform(_context) + text_transform(_question) + tok_ans
        vocab = build_vocab_from_iterator(apply_transform(raw_data['train']), len(raw_data['train']))
    text_transform = sequential_transforms(text_transform, vocab_func(vocab), totensor(dtype=torch.long))
    transforms = {'context': text_transform, 'question': text_transform,
                  'answers': text_transform, 'ans_pos': totensor(dtype=torch.long)}
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
    'SQuAD1': SQuAD1,
    'SQuAD2': SQuAD2
}
