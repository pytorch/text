import torch
import logging
import os
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.data.functional import read_text_iterator, create_data_from_iterator
from torchtext.datasets.text_classification import TextClassificationDataset

URLS = {
    'IMDB':
        'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
}


def _setup_datasets(dataset_name, tokenizer=get_tokenizer("basic_english"),
                    root='.data', vocab=None, removed_tokens=['<unk>']):
    dataset_tar = download_from_url(URLS[dataset_name], root=root)
    extracted_files = extract_archive(dataset_tar)
    for fname in extracted_files:
        if 'train' in fname:
            train_path = os.path.join(root, fname)
        if 'test' in fname:
            test_path = os.path.join(root, fname)

    if vocab is None:
        logging.info('Building Vocab based on {}'.format(train_path))
        vocab = build_vocab_from_iterator(read_text_iterator(train_path, tokenizer))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info('Vocab has {} entries'.format(len(vocab)))

    labels = ['pos', 'neg']

    logging.info('Creating training data')
    train_data = []
    for label in ['pos', 'neg']:
        for fname in glob.iglob(os.path.join(train_path, label, '*.txt')):
            text = list(create_data_from_iterator(vocab,
                                                  read_text_iterator(train_path,
                                                                     tokenizer),
                                                  removed_tokens))[0]
            train_data.append((torch.Tensor(text).long(), label))

    logging.info('Creating testing data')
    test_data = []
    for label in ['pos', 'neg']:
        for fname in glob.iglob(os.path.join(test_path, label, '*.txt')):
            text = list(create_data_from_iterator(vocab,
                                                  read_text_iterator(test_path,
                                                                     tokenizer),
                                                  removed_tokens))[0]
            test_data.append((torch.Tensor(text).long(), label))

    return (TextClassificationDataset(train_data, vocab, labels),
            TextClassificationDataset(test_data), vocab, labels)


def IMDB(*args, **kwargs):
    """ Defines WikiText2 datasets.
    Create language modeling dataset: WikiText2
    Separately returns the train/test/valid set
    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        removed_tokens: removed tokens from output dataset (Default: '<unk>')
    Examples:
        >>> from torchtext.datasets import WikiText2
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = WikiText2(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
    """

    return _setup_datasets(*(("IMDB",) + args), **kwargs)









import os
import glob
import io

from .. import data


class IMDB(data.Dataset):

    urls = ['http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz']
    name = 'imdb'
    dirname = 'aclImdb'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        """Create an IMDB dataset instance given a path and fields.

        Arguments:
            path: Path to the dataset's highest level directory
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        for label in ['pos', 'neg']:
            for fname in glob.iglob(os.path.join(path, label, '*.txt')):
                with io.open(fname, 'r', encoding="utf-8") as f:
                    text = f.readline()
                examples.append(data.Example.fromlist([text, label], fields))

        super(IMDB, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root='.data',
               train='train', test='test', **kwargs):
        """Create dataset objects for splits of the IMDB dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: Root dataset storage directory. Default is '.data'.
            train: The directory that contains the training examples
            test: The directory that contains the test examples
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        return super(IMDB, cls).splits(
            root=root, text_field=text_field, label_field=label_field,
            train=train, validation=None, test=test, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.data', vectors=None, **kwargs):
        """Create iterator objects for splits of the IMDB dataset.

        Arguments:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that contains the imdb dataset subdirectory
            vectors: one of the available pretrained vectors or a list with each
                element one of the available pretrained vectors (see Vocab.load_vectors)

            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, test), batch_size=batch_size, device=device)
