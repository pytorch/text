import torch
import logging
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

    if vocab is None:
        logging.info('Building Vocab based on train data')
        read_text = []
        for fname in extracted_files:
            if 'train' in fname and ('pos' in fname or 'neg' in fname):
                read_text += list(read_text_iterator(fname, tokenizer))
        vocab = build_vocab_from_iterator(read_text)
        torch.save(vocab, "imdb_vocab.pt")
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info('Vocab has {} entries'.format(len(vocab)))

    labels = {0, 1}
    logging.info('Creating train/test data')
    train_data = []
    test_data = []
    for fname in extracted_files:
        if 'urls' in fname:
            continue
        elif 'train' in fname:
            if 'pos' in fname:
                text = list(create_data_from_iterator(vocab,
                                                      read_text_iterator(fname,
                                                                         tokenizer),
                                                      removed_tokens))[0]
                train_data.append((1, torch.Tensor(text).long()))
            elif 'neg' in fname:
                text = list(create_data_from_iterator(vocab,
                                                      read_text_iterator(fname,
                                                                         tokenizer),
                                                      removed_tokens))[0]
                train_data.append((0, torch.Tensor(text).long()))
        elif 'test' in fname:
            if 'pos' in fname:
                text = list(create_data_from_iterator(vocab,
                                                      read_text_iterator(fname,
                                                                         tokenizer),
                                                      removed_tokens))[0]
                test_data.append((1, torch.Tensor(text).long()))
            elif 'neg' in fname:
                text = list(create_data_from_iterator(vocab,
                                                      read_text_iterator(fname,
                                                                         tokenizer),
                                                      removed_tokens))[0]
                test_data.append((0, torch.Tensor(text).long()))

    return (TextClassificationDataset(vocab, train_data, labels),
            TextClassificationDataset(vocab, test_data, labels))


def IMDB(*args, **kwargs):
    """ Defines IMDB datasets.
        The labels includes:
            - 0 : Negative
            - 1 : Positive

    Create sentiment analysis dataset: IMDB
    Separately returns the training and test dataset

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
        >>> from torchtext.datasets import IMDB
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset = IMDB(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
    """

    return _setup_datasets(*(("IMDB",) + args), **kwargs)
