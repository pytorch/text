import torch
from torch.nn import Sequential
import io
from torchtext.utils import unicode_csv_reader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.experimental.transforms import TokenizerTransform, NGrams, \
    VocabTransform, ToTensor
from .raw_text_classification import RawAG_NEWS, RawSogouNews, RawDBpedia, \
    RawYelpReviewPolarity, RawYelpReviewFull, RawYahooAnswers, \
    RawAmazonReviewPolarity, RawAmazonReviewFull


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

    def __init__(self, data, transforms):
        """Initiate text-classification dataset.
        Arguments:
        Examples:
        """

        super(TextClassificationDataset, self).__init__()
        self.data = data
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


def _setup_datasets(dataset_name, root='.data', ngrams=1, vocab=None):
    tok_transform = TokenizerTransform('basic_english')
    ngram_transform = NGrams(ngrams)
    processing_transform = Sequential(tok_transform, ngram_transform)

    train, test = DATASETS[dataset_name](root=root)
    if not vocab:
        vocab = build_vocab(train, processing_transform)
    label_transform = int
    text_transform = Sequential(processing_transform,
                                VocabTransform(vocab),
                                ToTensor(dtype=torch.long))
    return (TextClassificationDataset(train, (label_transform, text_transform)),
            TextClassificationDataset(test, (label_transform, text_transform)))


def AG_NEWS(*args, **kwargs):
    """ Defines AG_NEWS datasets.
        The labels includes:
            - 1 : World
            - 2 : Sports
            - 3 : Business
            - 4 : Sci/Tech

    Create supervised learning dataset: AG_NEWS

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
    """

    return _setup_datasets(*(('AG_NEWS',) + args), **kwargs)
#    return _setup_datasets(*(("AG_NEWS",) + args), **kwargs)


def SogouNews(*args, **kwargs):
    """ Defines SogouNews datasets.
        The labels includes:
            - 1 : Sports
            - 2 : Finance
            - 3 : Entertainment
            - 4 : Automobile
            - 5 : Technology
    Create supervised learning dataset: SogouNews
    Separately returns the training and test dataset
    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
    Examples:
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
    Create supervised learning dataset: DBpedia
    Separately returns the training and test dataset
    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
    Examples:
    """

    return _setup_datasets(*(("DBpedia",) + args), **kwargs)


def YelpReviewPolarity(*args, **kwargs):
    """ Defines YelpReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity.
            - 2 : Positive polarity.
    Create supervised learning dataset: YelpReviewPolarity
    Separately returns the training and test dataset
    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
    Examples:
    """

    return _setup_datasets(*(("YelpReviewPolarity",) + args), **kwargs)


def YelpReviewFull(*args, **kwargs):
    """ Defines YelpReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended).
    Create supervised learning dataset: YelpReviewFull
    Separately returns the training and test dataset
    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
    Examples:
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
    Create supervised learning dataset: YahooAnswers
    Separately returns the training and test dataset
    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
    Examples:
    """

    return _setup_datasets(*(("YahooAnswers",) + args), **kwargs)


def AmazonReviewPolarity(*args, **kwargs):
    """ Defines AmazonReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity
            - 2 : Positive polarity
    Create supervised learning dataset: AmazonReviewPolarity
    Separately returns the training and test dataset
    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
    Examples:
    """

    return _setup_datasets(*(("AmazonReviewPolarity",) + args), **kwargs)


def AmazonReviewFull(*args, **kwargs):
    """ Defines AmazonReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended)
    Create supervised learning dataset: AmazonReviewFull
    Separately returns the training and test dataset
    Arguments:
        root: Directory where the dataset are saved. Default: ".data"
    Examples:
    """

    return _setup_datasets(*(("AmazonReviewFull",) + args), **kwargs)


DATASETS = {
    'AG_NEWS': RawAG_NEWS,
    'SogouNews': RawSogouNews,
    'DBpedia': RawDBpedia,
    'YelpReviewPolarity': RawYelpReviewPolarity,
    'YelpReviewFull': RawYelpReviewFull,
    'YahooAnswers': RawYahooAnswers,
    'AmazonReviewPolarity': RawAmazonReviewPolarity,
    'AmazonReviewFull': RawAmazonReviewFull
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
                         5: 'score 5'}
}
