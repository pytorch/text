import torch
import logging
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext import datasets as raw
from torchtext.experimental.datasets.raw.common import check_default_set
from torchtext.experimental.datasets.raw.common import wrap_datasets
from torchtext.experimental.functional import (
    vocab_func,
    totensor,
    ngrams_func,
    sequential_transforms,
)

logger_ = logging.getLogger(__name__)


def build_vocab(data, transforms):
    def apply_transforms(data):
        for _, line in data:
            yield transforms(line)
    return build_vocab_from_iterator(apply_transforms(data), len(data))


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

        Args:
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
        label = self.data[i][0]
        txt = self.data[i][1]
        return (self.transforms[0](label), self.transforms[1](txt))

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        labels = []
        for item in self.data:
            label = item[0]
            labels.append(self.transforms[0](label))
        return set(labels)

    def get_vocab(self):
        return self.vocab


def _setup_datasets(dataset_name, root, ngrams, vocab, tokenizer, split_):
    text_transform = []
    if tokenizer is None:
        tokenizer = get_tokenizer("basic_english")
    text_transform = sequential_transforms(tokenizer, ngrams_func(ngrams))
    split = check_default_set(split_, ('train', 'test'), dataset_name)
    raw_datasets = raw.DATASETS[dataset_name](root=root, split=split)
    # Materialize raw text iterable dataset
    raw_data = {name: list(raw_dataset) for name, raw_dataset in zip(split, raw_datasets)}

    if vocab is None:
        if "train" not in split:
            raise TypeError("Must pass a vocab if train is not selected.")
        logger_.info('Building Vocab based on train data')
        vocab = build_vocab(raw_data["train"], text_transform)
    logger_.info('Vocab has %d entries', len(vocab))
    text_transform = sequential_transforms(
        text_transform, vocab_func(vocab), totensor(dtype=torch.long)
    )
    if dataset_name == 'IMDB':
        label_transform = sequential_transforms(lambda x: 1 if x == 'pos' else 0, totensor(dtype=torch.long))
    else:
        label_transform = sequential_transforms(totensor(dtype=torch.long))
    logger_.info('Building datasets for {}'.format(split))
    return wrap_datasets(tuple(
        TextClassificationDataset(
            raw_data[item], vocab, (label_transform, text_transform)
        )
        for item in split
    ), split_)


def AG_NEWS(root='.data', ngrams=1, vocab=None, tokenizer=None, split=('train', 'test')):
    """ Defines AG_NEWS datasets.
        The labels includes:
            - 1 : World
            - 2 : Sports
            - 3 : Business
            - 4 : Sci/Tech

    Create text classification dataset: AG_NEWS

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        split: a string or tuple for the returned datasets
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
        >>> train = AG_NEWS(tokenizer=tokenizer, split='train')

    """

    return _setup_datasets("AG_NEWS", root, ngrams, vocab, tokenizer, split)


def SogouNews(root='.data', ngrams=1, vocab=None, tokenizer=None, split=('train', 'test')):
    """ Defines SogouNews datasets.
        The labels includes:
            - 1 : Sports
            - 2 : Finance
            - 3 : Entertainment
            - 4 : Automobile
            - 5 : Technology

    Create text classification dataset: SogouNews

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        split: a string or tuple for the returned datasets
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
        >>> train = SogouNews(tokenizer=tokenizer, split='train')

    """

    return _setup_datasets("SogouNews", root, ngrams, vocab, tokenizer, split)


def DBpedia(root='.data', ngrams=1, vocab=None, tokenizer=None, split=('train', 'test')):
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

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        split: a string or tuple for the returned datasets
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
        >>> train = DBpedia(tokenizer=tokenizer, split='train')

    """

    return _setup_datasets("DBpedia", root, ngrams, vocab, tokenizer, split)


def YelpReviewPolarity(root='.data', ngrams=1, vocab=None, tokenizer=None, split=('train', 'test')):
    """ Defines YelpReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity.
            - 2 : Positive polarity.

    Create text classification dataset: YelpReviewPolarity

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        split: a string or tuple for the returned datasets
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
        >>> train = YelpReviewPolarity(tokenizer=tokenizer, split='train')

    """

    return _setup_datasets("YelpReviewPolarity", root, ngrams, vocab, tokenizer, split)


def YelpReviewFull(root='.data', ngrams=1, vocab=None, tokenizer=None, split=('train', 'test')):
    """ Defines YelpReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended).

    Create text classification dataset: YelpReviewFull

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        split: a string or tuple for the returned datasets
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
        >>> train = YelpReviewFull(tokenizer=tokenizer, split='train')

    """

    return _setup_datasets("YelpReviewFull", root, ngrams, vocab, tokenizer, split)


def YahooAnswers(root='.data', ngrams=1, vocab=None, tokenizer=None, split=('train', 'test')):
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

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        split: a string or tuple for the returned datasets
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
        >>> train = YahooAnswers(tokenizer=tokenizer, split='train')

    """

    return _setup_datasets("YahooAnswers", root, ngrams, vocab, tokenizer, split)


def AmazonReviewPolarity(root='.data', ngrams=1, vocab=None, tokenizer=None, split=('train', 'test')):
    """ Defines AmazonReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity
            - 2 : Positive polarity

    Create text classification dataset: AmazonReviewPolarity

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        split: a string or tuple for the returned datasets
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
        >>> train = AmazonReviewPolarity(tokenizer=tokenizer, split='train')

    """

    return _setup_datasets("AmazonReviewPolarity", root, ngrams, vocab, tokenizer, split)


def AmazonReviewFull(root='.data', ngrams=1, vocab=None, tokenizer=None, split=('train', 'test')):
    """ Defines AmazonReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended)

    Create text classification dataset: AmazonReviewFull

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        split: a string or tuple for the returned datasets
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
        >>> train = AmazonReviewFull(tokenizer=tokenizer, split='train')

    """

    return _setup_datasets("AmazonReviewFull", root, ngrams, vocab, tokenizer, split)


def IMDB(root='.data', ngrams=1, vocab=None, tokenizer=None, split=('train', 'test')):
    """ Defines IMDB datasets.
        The labels includes:
            - 0 : Negative
            - 1 : Positive

    Create sentiment analysis dataset: IMDB

    Separately returns the training and test dataset

    Args:
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
        split: a string or tuple for the returned datasets
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
        >>> train = IMDB(tokenizer=tokenizer, split='train')

    """

    return _setup_datasets("IMDB", root, ngrams, vocab, tokenizer, split)


DATASETS = {
    "AG_NEWS": AG_NEWS,
    "SogouNews": SogouNews,
    "DBpedia": DBpedia,
    "YelpReviewPolarity": YelpReviewPolarity,
    "YelpReviewFull": YelpReviewFull,
    "YahooAnswers": YahooAnswers,
    "AmazonReviewPolarity": AmazonReviewPolarity,
    "AmazonReviewFull": AmazonReviewFull,
    "IMDB": IMDB,
}


LABELS = {
    "AG_NEWS": {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"},
    "SogouNews": {
        1: "Sports",
        2: "Finance",
        3: "Entertainment",
        4: "Automobile",
        5: "Technology",
    },
    "DBpedia": {
        1: "Company",
        2: "EducationalInstitution",
        3: "Artist",
        4: "Athlete",
        5: "OfficeHolder",
        6: "MeanOfTransportation",
        7: "Building",
        8: "NaturalPlace",
        9: "Village",
        10: "Animal",
        11: "Plant",
        12: "Album",
        13: "Film",
        14: "WrittenWork",
    },
    "YelpReviewPolarity": {1: "Negative polarity", 2: "Positive polarity"},
    "YelpReviewFull": {
        1: "score 1",
        2: "score 2",
        3: "score 3",
        4: "score 4",
        5: "score 5",
    },
    "YahooAnswers": {
        1: "Society & Culture",
        2: "Science & Mathematics",
        3: "Health",
        4: "Education & Reference",
        5: "Computers & Internet",
        6: "Sports",
        7: "Business & Finance",
        8: "Entertainment & Music",
        9: "Family & Relationships",
        10: "Politics & Government",
    },
    "AmazonReviewPolarity": {1: "Negative polarity", 2: "Positive polarity"},
    "AmazonReviewFull": {
        1: "score 1",
        2: "score 2",
        3: "score 3",
        4: "score 4",
        5: "score 5",
    },
    "IMDB": {0: "Negative", 1: "Positive"},
}
