import os
import re
import torch
from ..utils import download_and_extract_archive, makedir_exist_ok
import random
from .. import data
from tqdm import tqdm
from ..data import dataset


class TextClassificationDataset(data.Dataset):
    """Defines text classification datasets.
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

    def __init__(self, dataset_name, root='.data',
                 text_field=None, label_field=None, ngrams=1):
        """Create a text classification dataset instance.

        Arguments:
            dataset_name: The name of dataset, include "ag_news", "sogou_news",
                "dbpedia", "yelp_review_polarity", "yelp_review_full",
                "yahoo_answers", "amazon_review_full", "amazon_review_polarity".
            root: Directory where the dataset are saved. Default: ".data"
            text_field: The field that will be used for the sentence. If not given,
                'spacy' token will be used.
            label_field: The field that will be used for the label. If not given,
                'float' token will be used.
            ngrams: a contiguous sequence of n items from s string text.
                Default: 1

        Examples:
            >>> txt_cls = TextClassificationDataset("ag_news", ngrams=2)

        """
        self.dataset_name = dataset_name
        if dataset_name in supported_text_classification_datasets.keys():
            self.url = supported_text_classification_datasets[dataset_name]
        else:
            raise ValueError("dataset_name %s is not supported." % dataset_name)
        self.root = root
        fields = []
        fields.append(('text', text_field if text_field is not None
                       else data.Field(tokenize=data.get_tokenizer('spacy'),
                                       init_token='<SOS>',
                                       eos_token='<EOS>')))
        fields.append(('label', label_field if label_field is not None
                       else data.LabelField(dtype=torch.float)))
        examples = []
        super(TextClassificationDataset, self).__init__(examples, fields)

        self.raw_folder = os.path.join(self.root, self.__class__.__name__, 'raw')
        self.processed_folder = os.path.join(self.root,
                                             self.__class__.__name__,
                                             'processed')

        self.train_examples = self.load_train_data(ngrams)
        self.test_examples = self.load_test_data(ngrams)
        self.examples = self.train_examples + self.test_examples

        self.fields['text'].build_vocab(self)
        self.fields['label'].build_vocab(self)

    def splits(self):
        """ Create dataset objects for splits of the dataset.
        """
        raise NotImplementedError

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def split(self, split_ratio=0.7, random_state=None):
        """ Create train-test-valid splits from the instance's examples.
            The initial dataset has no valid set. Therefore, the train set
            is split into train and valid parts.

        Inputs:
            split_ratio: the percent out of the initial train set split into
                train dataset objects. Default: 0.7

        Examples:
            >>> train_data, test_data, valid_data = txt_cls.split()

        Outputs:
            - train_data: a datset objects based on
                train.csv file with split_ratio percent.
            - train_data: a datset objects based on test.csv file.
            - valid_data: a datset objects based on
                train.csv file with 1-split_ratio percent.

        """
        rnd = dataset.RandomShuffler(random_state)
        train_examples, test_examples, valid_examples = \
            dataset.rationed_split(self.train_examples,
                                   split_ratio, 0.0,
                                   1 - split_ratio, rnd)
        test_examples = self.test_examples
        split = tuple(data.Dataset(d, self.fields)
                      for d in (train_examples, test_examples, valid_examples)
                      if d is not None)
        for subset in split:
            subset.sort_key = self.sort_key

        return split

    def iters(self, split_ratio=0.7, batch_size=32, device='cpu', random_state=None):
        """Create iterator objects for splits of the dataset.

        Arguments:
            split_ratio: split train_examples into train set (split_ratio)
                and valid set (1-split_ratio). Default: 0.7
            batch_size: batch size. Default: 32
            device: the device to sent data. Default: 'cpu'
            random_state: the random state provided by user. Default: None

        Examples:
            >>> train_iter, test_iter, valid_iter = txt_cls.iters(device="cpu")

        Outputs:
            - train_iter: a iterator based on
                train.csv file with split_ratio percent.
            - train_iter: a iterator based on test.csv file.
            - valid_iter: a iterator based on
                train.csv file with 1-split_ratio percent.

        """

        train, test, valid = self.split(split_ratio, random_state)
        return data.BucketIterator.splits(
            (train, test, valid), batch_size=batch_size, device=device)

    def download(self):
        """Download the dataset if it doesn't exist in processed_folder already."""

        train_csv_path = os.path.join(self.raw_folder,
                                      self.dataset_name + '_csv',
                                      'train.csv')
        test_csv_path = os.path.join(self.raw_folder,
                                     self.dataset_name + '_csv',
                                     'test.csv')
        if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
            return

        makedir_exist_ok(self.raw_folder)
        filename = self.dataset_name + '_csv.tar.gz'
        if self.dataset_name in ['ag_news', 'dbpedia']:
            download_and_extract_archive(self.url, download_root=self.raw_folder,
                                         extract_root=self.raw_folder, filename=filename,
                                         remove_finished=True)
        else:
            os.system("curl -c ./cookies \"%s\" > ./intermezzo.html" % self.url)
            os.system("curl -L -b ./cookies \"https:" +
                      "//drive.google.com$(cat ./intermezzo.html " +
                      "| grep -Po \'uc-download-link\" [^>]* " +
                      "href=\"" + r"\K" + "[^\"]*\' " +
                      "| sed \'s/" + r"\&" + "amp;/" + r"\&" +
                      "/g\')\" > \"%s\" " % str(self.raw_folder + '/' + filename))
            os.system("tar -xzvf \"%s\" -C \"%s\"" % (self.raw_folder +
                      '/' + filename, self.raw_folder))
            os.system("rm ./cookies ./intermezzo.html")

        print('Dataset %s downloaded.' % self.dataset_name)

    def preprocess(self, raw_folder=None, processed_folder=None):
        """Preprocess the csv files."""

        if raw_folder is None:
            raw_folder = os.path.join(self.raw_folder, self.dataset_name + '_csv')

        if processed_folder is None:
            processed_folder = self.processed_folder

        if os.path.exists(processed_folder) is not True:
            makedir_exist_ok(processed_folder)

        src_filepath = os.path.join(raw_folder, 'train.csv')
        if not os.path.isfile(src_filepath):
            self.download()
        tgt_filepath = os.path.join(processed_folder, self.dataset_name + '.train')
        self.text_normalize(src_filepath, tgt_filepath)

        src_filepath = os.path.join(raw_folder, 'test.csv')
        if not os.path.isfile(src_filepath):
            self.download()
        tgt_filepath = os.path.join(processed_folder, self.dataset_name + '.test')
        self.text_normalize(src_filepath, tgt_filepath)

        print("Dataset %s preprocessed." % self.dataset_name)

    def text_normalize(self, src_filepath, tgt_filepath):
        """Normalize text and separate label/text."""

        lines = []
        with open(src_filepath) as src_data:
            with open(tgt_filepath, 'w') as new_data:
                for line in src_data:
                    line = line.lower()
                    label, text = line.split(",", 1)
                    label = "__label__" + re.sub(r'[^0-9\s]', '', label)
                    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
                    text = ' '.join(text.split())
                    line = label + ' , ' + text + ' \n'
                    lines.append(line)
                random.shuffle(lines)
                new_data.writelines(lines)
        return

    @classmethod
    def generate_ngrams(cls, token_list, num):
        """Generate a list of token with ngrams.

        Arguments:
            token_list: A list of tokens
            num: the number of ngrams.

        Examples:
            >>> token_list = ['here', 'we', 'are']
            >>> print(TextClassificationDataset.generate_ngrams(token_list, 2))
            >>> ['here we', 'we are']
        """

        sequences = [token_list[i:] for i in range(num)]
        ngram_list = list(zip(*sequences))
        return [' '.join(list(gram)) for gram in ngram_list]

    def label_text_processor(self, line, ngrams=1):
        """Process text string and generate examples for dataset."""
        fields = [('text', self.fields['text']), ('label', self.fields['label'])]
        label, text = line.split(",", 1)
        label = float(label.split("__label__")[1])
        ex = data.Example.fromlist([text, label], fields)
        tokens = ex.text[1:]  # Skip the first space '\t'
        ex.text = []
        for n in range(1, ngrams + 1):
            ex.text += self.generate_ngrams(tokens, n)
        return ex

    def _load_text_classification_data(self, filepath, ngrams=1):
        examples = []
        with open(filepath) as src_data:
            for line in tqdm(src_data):
                examples.append(self.label_text_processor(line, ngrams))
        return examples

    def load_train_data(self, ngrams=1):
        """Load train data and generate ngrams."""
        filepath = os.path.join(self.processed_folder, self.dataset_name + '.train')

        if not os.path.isfile(filepath):
            self.preprocess()

        return self._load_text_classification_data(filepath, ngrams)

    def load_test_data(self, ngrams=1):
        """Load test data and generate ngrams."""
        filepath = os.path.join(self.processed_folder, self.dataset_name + '.test')

        if not os.path.isfile(filepath):
            self.preprocess()

        return self._load_text_classification_data(filepath, ngrams)


class AG_NEWS(TextClassificationDataset):
    """ Defines AG_NEWS datasets.
        The labels includes:
            - 1 : World
            - 2 : Sports
            - 3 : Business
            - 4 : Sci/Tech
     """
    def __init__(self, root='.data', text_field=None, label_field=None, ngrams=1):
        """Create supervised learning dataset: AG_NEWS

        Inputs:
            See TextClassificationDataset() class

        Examples:
            >>> text_cls = torchtext.datasets.AG_NEWS(ngrams=3)
            >>> train_iter, test_iter, valid_iter = txt_cls.iters(device="cpu")

        """
        super(AG_NEWS, self).__init__('ag_news', root, text_field,
                                      label_field, ngrams)


class SogouNews(TextClassificationDataset):
    """ Defines SogouNews datasets.
        The labels includes:
            - 1 : Sports
            - 2 : Finance
            - 3 : Entertainment
            - 4 : Automobile
            - 5 : Technology
     """
    def __init__(self, root='.data', text_field=None,
                 label_field=None, ngrams=1):
        """Create supervised learning dataset: SogouNews

        Inputs:
            See TextClassificationDataset() class

        Examples:
            >>> text_cls = torchtext.datasets.SogouNews(ngrams=3)
            >>> train_iter, test_iter, valid_iter = txt_cls.iters(device="cpu")

        """
        super(SogouNews, self).__init__('sogou_news', root, text_field,
                                        label_field, ngrams)


class DBpedia(TextClassificationDataset):
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
     """
    def __init__(self, root='.data', text_field=None, label_field=None, ngrams=1):
        """Create supervised learning dataset: DBpedia

        Inputs:
            See TextClassificationDataset() class

        Examples:
            >>> text_cls = torchtext.datasets.DBpedia(ngrams=3)
            >>> train_iter, test_iter, valid_iter = txt_cls.iters(device="cpu")

        """
        super(DBpedia, self).__init__('dbpedia', root, text_field, label_field, ngrams)


class YelpReviewPolarity(TextClassificationDataset):
    """ Defines YelpReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity.
            - 2 : Positive polarity.
     """
    def __init__(self, root='.data', text_field=None, label_field=None, ngrams=1):
        """Create supervised learning dataset: YelpReviewPolarity

        Inputs:
            See TextClassificationDataset() class

        Examples:
            >>> text_cls = torchtext.datasets.YelpReviewPolarity(ngrams=3)
            >>> train_iter, test_iter, valid_iter = txt_cls.iters(device="cpu")

        """
        super(YelpReviewPolarity, self).__init__('yelp_review_polarity',
                                                 root, text_field, label_field, ngrams)


class YelpReviewFull(TextClassificationDataset):
    """ Defines YelpReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended).
     """
    def __init__(self, root='.data', text_field=None, label_field=None, ngrams=1):
        """Create supervised learning dataset: YelpReviewFull

        Inputs:
            See TextClassificationDataset() class

        Examples:
            >>> text_cls = torchtext.datasets.YelpReviewFull(ngrams=3)
            >>> train_iter, test_iter, valid_iter = txt_cls.iters(device="cpu")

        """
        super(YelpReviewFull, self).__init__('yelp_review_full',
                                             root, text_field, label_field, ngrams)


class YahooAnswers(TextClassificationDataset):
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
     """
    def __init__(self, root='.data', text_field=None, label_field=None, ngrams=1):
        """Create supervised learning dataset: YahooAnswers

        Inputs:
            See TextClassificationDataset() class

        Examples:
            >>> text_cls = torchtext.datasets.YahooAnswers(ngrams=3)
            >>> train_iter, test_iter, valid_iter = txt_cls.iters(device="cpu")

        """
        super(YahooAnswers, self).__init__('yahoo_answers',
                                           root, text_field, label_field, ngrams)


class AmazonReviewPolarity(TextClassificationDataset):
    """ Defines AmazonReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity
            - 2 : Positive polarity
     """
    def __init__(self, root='.data', text_field=None,
                 label_field=None, ngrams=1):
        """Create supervised learning dataset: AmazonReviewPolarity

        Inputs:
            See TextClassificationDataset() class

        Examples:
            >>> text_cls = torchtext.datasets.AmazonReviewPolarity(ngrams=3)
            >>> train_iter, test_iter, valid_iter = txt_cls.iters(device="cpu")

        """
        super(AmazonReviewPolarity, self).__init__('amazon_review_polarity',
                                                   root, text_field, label_field, ngrams)


class AmazonReviewFull(TextClassificationDataset):
    """ Defines AmazonReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended)
     """
    def __init__(self, root='.data', text_field=None, label_field=None, ngrams=1):
        """Create supervised learning dataset: AmazonReviewFull

        Inputs:
            See TextClassificationDataset() class

        Examples:
            >>> text_cls = torchtext.datasets.AmazonReviewFull(ngrams=3)
            >>> train_iter, test_iter, valid_iter = txt_cls.iters(device="cpu")

        """
        super(AmazonReviewFull, self).__init__('amazon_review_full',
                                               root, text_field, label_field, ngrams)


supported_text_classification_datasets = {
    'ag_news':
    'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms',
    'sogou_news':
    'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE',
    'dbpedia':
    'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k',
    'yelp_review_polarity':
    'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg',
    'yelp_review_full':
    'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0',
    'yahoo_answers':
    'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU',
    'amazon_review_full':
    'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA',
    'amazon_review_polarity':
    'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM'
}
