import torch
import io
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.vocab import build_vocab_from_iterator

URLS = {
    'AG_NEWS':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms',
    'SogouNews':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE',
    'DBpedia':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k',
    'YelpReviewPolarity':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg',
    'YelpReviewFull':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0',
    'YahooAnswers':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU',
    'AmazonReviewPolarity':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM',
    'AmazonReviewFull':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA'
}


def _create_data_from_csv(data_path):
    data = []
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            data.append((row[0], ' '.join(row[1:])))
    return data


def build_vocab(txt_iter, transform):
    tok_list = [transform(seq) for seq in txt_iter]
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
            vocab: Vocabulary object used for dataset.
        Examples:
            See the examples in examples/text_classification/
        """

        super(TextClassificationDataset, self).__init__()
        self.data = data
        self.transforms = transforms  # (label_transforms, tokens_transforms)

    def __getitem__(self, i):
        return (self.transforms[0](self.data[i][0]),
                self.transforms[1](self.data[i][1]))

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return set([self.transforms[0](item[0]) for item in self.data])


def _setup_datasets(dataset_name, root='.data', transforms=[int, lambda x:x]):
    dataset_tar = download_from_url(URLS[dataset_name], root=root)
    extracted_files = extract_archive(dataset_tar)

    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname

    train_data = _create_data_from_csv(train_csv_path)
    test_data = _create_data_from_csv(test_csv_path)
    return (TextClassificationDataset(train_data, transforms),
            TextClassificationDataset(test_data, transforms))


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

    return _setup_datasets(*(("AG_NEWS",) + args), **kwargs)


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
    'AG_NEWS': AG_NEWS,
    'SogouNews': SogouNews,
    'DBpedia': DBpedia,
    'YelpReviewPolarity': YelpReviewPolarity,
    'YelpReviewFull': YelpReviewFull,
    'YahooAnswers': YahooAnswers,
    'AmazonReviewPolarity': AmazonReviewPolarity,
    'AmazonReviewFull': AmazonReviewFull
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
