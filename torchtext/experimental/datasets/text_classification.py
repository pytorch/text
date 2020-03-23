import torch
import io
from torchtext.utils import unicode_csv_reader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.experimental.datasets.raw import AG_NEWS as RawAG_NEWS
from torchtext.experimental.datasets.raw import SogouNews as RawSogouNews
from torchtext.experimental.datasets.raw import DBpedia as RawDBpedia
from torchtext.experimental.datasets.raw import YelpReviewPolarity as \
    RawYelpReviewPolarity
from torchtext.experimental.datasets.raw import YelpReviewFull as RawYelpReviewFull
from torchtext.experimental.datasets.raw import YahooAnswers as RawYahooAnswers
from torchtext.experimental.datasets.raw import AmazonReviewPolarity as \
    RawAmazonReviewPolarity
from torchtext.experimental.datasets.raw import AmazonReviewFull as RawAmazonReviewFull
from torchtext.experimental.datasets.raw import IMDB as RawIMDB


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
        return [self.vocab[tok] for tok in tok_iter]


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


def _create_data_from_csv(data_path):
    data = []
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            data.append((row[0], ' '.join(row[1:])))
    return data


def build_vocab(dataset, transform):
    # if not isinstance(dataset, TextClassificationDataset):
    #   raise TypeError('Passed dataset is not TextClassificationDataset')

    # data are saved in the form of (label, text_string)
    tok_list = [transform(seq[1]) for seq in dataset.data]
    return build_vocab_from_iterator(tok_list)


class TextClassificationDataset(torch.utils.data.Dataset):
    """Defines an abstract text classification datasets.
       Currently, we only support the following datasets:
             - AG_NEWS
             - SogouNews
             - DBpedia
             - YelpReviewPolarity
             - YelpReviewFull
             - YahooAnswers
             - AmazonReviewPolarity
             - AmazonReviewFull
    """

    def __init__(self, data, vocab, transforms):
        """Initiate text-classification dataset.

        Arguments:
            data: a list of label and text tring tuple. label is an integer.
                [(label1, text1), (label2, text2), (label2, text3)]
            vocab: Vocabulary object used for dataset.
            transforms: a tuple of label and text string transforms.
        """

        super(TextClassificationDataset, self).__init__()
        self.data = data
        self.vocab = vocab
        self.transforms = transforms  # (label_transforms, tokens_transforms)

    def __getitem__(self, i):
        txt = self.data[i][1]
        for transform in self.transforms[1]:
            txt = transform(txt)
        return (self.transforms[0](self.data[i][0]), txt)

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return set([self.transforms[0](item[0]) for item in self.data])

    def get_vocab(self):
        return self.vocab


def _setup_datasets(dataset_name, root='.data', ngrams=1, vocab=None,
                    tokenizer=None, data_select=('train', 'test')):
    if not tokenizer:
        tok_transform = TokenizerTransform()
    else:
        tok_transform = TokenizerTransform(tokenizer)

    if not set(data_select).issubset(set(('train', 'test'))):
        raise TypeError('Given data selection {} is not supported!'.format(data_select))

    ngram_transform = NGrams(ngrams)
    processing_transform = TextSequential(tok_transform, ngram_transform)

    train, test = DATASETS[dataset_name](root=root)
    raw_data = {'train': train,
                'test': test}

    if not vocab:
        if 'train' not in data_select:
            raise TypeError("Must pass a vocab if train is not selected.")
        vocab = build_vocab(train, processing_transform)
    label_transform = ToTensor(dtype=torch.long)
    text_transform = TextSequential(processing_transform,
                                    VocabTransform(vocab),
                                    ToTensor(dtype=torch.long))
    return tuple(TextClassificationDataset(raw_data[item], vocab,
                                           (label_transform, text_transform))
                 for item in data_select)


def AG_NEWS(*args, **kwargs):
    """ Defines AG_NEWS datasets.
        The labels includes:
            - 1 : World
            - 2 : Sports
            - 3 : Business
            - 4 : Sci/Tech

    Create text classification dataset: AG_NEWS

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import AG_NEWS
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = AG_NEWS(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = AG_NEWS(tokenizer=tokenizer)
        >>> train, = AG_NEWS(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(('AG_NEWS',) + args), **kwargs)


def SogouNews(*args, **kwargs):
    """ Defines SogouNews datasets.
        The labels includes:
            - 1 : Sports
            - 2 : Finance
            - 3 : Entertainment
            - 4 : Automobile
            - 5 : Technology

    Create text classification dataset: SogouNews

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import SogouNews
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = SogouNews(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = SogouNews(tokenizer=tokenizer)
        >>> train, = SogouNews(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("SogouNews",) + args), **kwargs)


def DBpedia(*args, **kwargs):
    """ Defines DBpedia datasets.
        The labels includes:
            - 1 : Company
            - 2 : EducationalInstitution
            - 3 : Artist
            - 4 : Athlete
            - 5 : OfficeHolder
            - 6 : MeanOfTransportation
            - 7 : Building
            - 8 : NaturalPlace
            - 9 : Village
            - 10 : Animal
            - 11 : Plant
            - 12 : Album
            - 13 : Film
            - 14 : WrittenWork

    Create text classification dataset: DBpedia

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import DBpedia
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = DBpedia(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = DBpedia(tokenizer=tokenizer)
        >>> train, = DBpedia(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("DBpedia",) + args), **kwargs)


def YelpReviewPolarity(*args, **kwargs):
    """ Defines YelpReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity.
            - 2 : Positive polarity.

    Create text classification dataset: YelpReviewPolarity

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import YelpReviewPolarity
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = YelpReviewPolarity(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = YelpReviewPolarity(tokenizer=tokenizer)
        >>> train, = YelpReviewPolarity(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("YelpReviewPolarity",) + args), **kwargs)


def YelpReviewFull(*args, **kwargs):
    """ Defines YelpReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended).

    Create text classification dataset: YelpReviewFull

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import YelpReviewFull
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = YelpReviewFull(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = YelpReviewFull(tokenizer=tokenizer)
        >>> train, = YelpReviewFull(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("YelpReviewFull",) + args), **kwargs)


def YahooAnswers(*args, **kwargs):
    """ Defines YahooAnswers datasets.
        The labels includes:
            - 1 : Society & Culture
            - 2 : Science & Mathematics
            - 3 : Health
            - 4 : Education & Reference
            - 5 : Computers & Internet
            - 6 : Sports
            - 7 : Business & Finance
            - 8 : Entertainment & Music
            - 9 : Family & Relationships
            - 10 : Politics & Government

    Create text classification dataset: YahooAnswers

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import YahooAnswers
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = YahooAnswers(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = YahooAnswers(tokenizer=tokenizer)
        >>> train, = YahooAnswers(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("YahooAnswers",) + args), **kwargs)


def AmazonReviewPolarity(*args, **kwargs):
    """ Defines AmazonReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity
            - 2 : Positive polarity

    Create text classification dataset: AmazonReviewPolarity

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import AmazonReviewPolarity
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = AmazonReviewPolarity(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = AmazonReviewPolarity(tokenizer=tokenizer)
        >>> train, = AmazonReviewPolarity(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("AmazonReviewPolarity",) + args), **kwargs)


def AmazonReviewFull(*args, **kwargs):
    """ Defines AmazonReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended)

    Create text classification dataset: AmazonReviewFull

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import AmazonReviewFull
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = AmazonReviewFull(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = AmazonReviewFull(tokenizer=tokenizer)
        >>> train, = AmazonReviewFull(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("AmazonReviewFull",) + args), **kwargs)


def IMDB(*args, **kwargs):
    """ Defines IMDB datasets.
        The labels includes:
            - 0 : Negative
            - 1 : Positive

    Create sentiment analysis dataset: IMDB

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        removed_tokens: removed tokens from output dataset (Default: [])
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import IMDB
        >>> from torchtext.data.utils import get_tokenizer
        >>> train, test = IMDB(ngrams=3)
        >>> tokenizer = get_tokenizer("spacy")
        >>> train, test = IMDB(tokenizer=tokenizer)
        >>> train, = IMDB(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("IMDB",) + args), **kwargs)


DATASETS = {
    'AG_NEWS': RawAG_NEWS,
    'SogouNews': RawSogouNews,
    'DBpedia': RawDBpedia,
    'YelpReviewPolarity': RawYelpReviewPolarity,
    'YelpReviewFull': RawYelpReviewFull,
    'YahooAnswers': RawYahooAnswers,
    'AmazonReviewPolarity': RawAmazonReviewPolarity,
    'AmazonReviewFull': RawAmazonReviewFull,
    'IMDB': RawIMDB
}


LABELS = {
    'AG_NEWS': {1: 'World',
                2: 'Sports',
                3: 'Business',
                4: 'Sci/Tech'},
    'SogouNews': {1: 'Sports',
                  2: 'Finance',
                  3: 'Entertainment',
                  4: 'Automobile',
                  5: 'Technology'},
    'DBpedia': {1: 'Company',
                2: 'EducationalInstitution',
                3: 'Artist',
                4: 'Athlete',
                5: 'OfficeHolder',
                6: 'MeanOfTransportation',
                7: 'Building',
                8: 'NaturalPlace',
                9: 'Village',
                10: 'Animal',
                11: 'Plant',
                12: 'Album',
                13: 'Film',
                14: 'WrittenWork'},
    'YelpReviewPolarity': {1: 'Negative polarity',
                           2: 'Positive polarity'},
    'YelpReviewFull': {1: 'score 1',
                       2: 'score 2',
                       3: 'score 3',
                       4: 'score 4',
                       5: 'score 5'},
    'YahooAnswers': {1: 'Society & Culture',
                     2: 'Science & Mathematics',
                     3: 'Health',
                     4: 'Education & Reference',
                     5: 'Computers & Internet',
                     6: 'Sports',
                     7: 'Business & Finance',
                     8: 'Entertainment & Music',
                     9: 'Family & Relationships',
                     10: 'Politics & Government'},
    'AmazonReviewPolarity': {1: 'Negative polarity',
                             2: 'Positive polarity'},
    'AmazonReviewFull': {1: 'score 1',
                         2: 'score 2',
                         3: 'score 3',
                         4: 'score 4',
                         5: 'score 5'},
    'IMDB': {0: 'Negative',
             1: 'Positive'}
}
