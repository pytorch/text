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
