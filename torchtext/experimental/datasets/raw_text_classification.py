import torch
import io
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader

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


class RawTextDataset(torch.utils.data.Dataset):
    """Defines an abstraction for raw text datasets.
    """

    def __init__(self, data):
        """Initiate text-classification dataset.
        Arguments:
        Examples:
        """

        super(RawTextDataset, self).__init__()
        self.data = data

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


def _setup_datasets(dataset_name, root='.data'):
    dataset_tar = download_from_url(URLS[dataset_name], root=root)
    extracted_files = extract_archive(dataset_tar)

    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname

    train_data = _create_data_from_csv(train_csv_path)
    test_data = _create_data_from_csv(test_csv_path)
    return (RawTextDataset(train_data),
            RawTextDataset(test_data))


def RawAG_NEWS(*args, **kwargs):
    """ Defines AG_NEWS datasets.
    """

    return _setup_datasets(*(("AG_NEWS",) + args), **kwargs)


def RawSogouNews(*args, **kwargs):
    """ Defines SogouNews datasets.
    """

    return _setup_datasets(*(("SogouNews",) + args), **kwargs)


def RawDBpedia(*args, **kwargs):
    """ Defines DBpedia datasets.
    """

    return _setup_datasets(*(("DBpedia",) + args), **kwargs)


def RawYelpReviewPolarity(*args, **kwargs):
    """ Defines YelpReviewPolarity datasets.
    """

    return _setup_datasets(*(("YelpReviewPolarity",) + args), **kwargs)


def RawYelpReviewFull(*args, **kwargs):
    """ Defines YelpReviewFull datasets.
    """

    return _setup_datasets(*(("YelpReviewFull",) + args), **kwargs)


def RawYahooAnswers(*args, **kwargs):
    """ Defines YahooAnswers datasets.
    """

    return _setup_datasets(*(("YahooAnswers",) + args), **kwargs)


def RawAmazonReviewPolarity(*args, **kwargs):
    """ Defines AmazonReviewPolarity datasets.
    """

    return _setup_datasets(*(("AmazonReviewPolarity",) + args), **kwargs)


def RawAmazonReviewFull(*args, **kwargs):
    """ Defines AmazonReviewFull datasets.
    """

    return _setup_datasets(*(("AmazonReviewFull",) + args), **kwargs)


DATASETS = {
    'RawAG_NEWS': RawAG_NEWS,
    'RawSogouNews': RawSogouNews,
    'RawDBpedia': RawDBpedia,
    'RawYelpReviewPolarity': RawYelpReviewPolarity,
    'RawYelpReviewFull': RawYelpReviewFull,
    'RawYahooAnswers': RawYahooAnswers,
    'RawAmazonReviewPolarity': RawAmazonReviewPolarity,
    'RawAmazonReviewFull': RawAmazonReviewFull
}


LABELS = {
    'RawAG_NEWS': {1: 'World',
                   2: 'Sports',
                   3: 'Business',
                   4: 'Sci/Tech'},
    'RawSogouNews': {1: 'Sports',
                     2: 'Finance',
                     3: 'Entertainment',
                     4: 'Automobile',
                     5: 'Technology'},
    'RawDBpedia': {1: 'Company',
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
    'RawYelpReviewPolarity': {1: 'Negative polarity',
                              2: 'Positive polarity'},
    'RawYelpReviewFull': {1: 'score 1',
                          2: 'score 2',
                          3: 'score 3',
                          4: 'score 4',
                          5: 'score 5'},
    'RawYahooAnswers': {1: 'Society & Culture',
                        2: 'Science & Mathematics',
                        3: 'Health',
                        4: 'Education & Reference',
                        5: 'Computers & Internet',
                        6: 'Sports',
                        7: 'Business & Finance',
                        8: 'Entertainment & Music',
                        9: 'Family & Relationships',
                        10: 'Politics & Government'},
    'RawAmazonReviewPolarity': {1: 'Negative polarity',
                                2: 'Positive polarity'},
    'RawAmazonReviewFull': {1: 'score 1',
                            2: 'score 2',
                            3: 'score 3',
                            4: 'score 4',
                            5: 'score 5'}
}
