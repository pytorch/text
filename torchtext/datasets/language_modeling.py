import torch
import logging
import io
import os
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from tqdm import tqdm

URLS = {
    'WikiText2':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
    'WikiText103':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip',
    'PennTreebank':
        ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt']
}


def _read_text_iterator(data_path, tokenizer):
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            tokens = tokenizer(' '.join(row))
            yield tokens


def _create_data_from_iterator(vocab, iterator, include_unk):
    _data = []
    with tqdm(unit_scale=0, unit='lines') as t:
        for tokens in iterator:
            if include_unk:
                tokens = [vocab[token] for token in tokens]
            else:
                token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                        for token in tokens]))
                tokens = token_ids
            if len(tokens) == 0:
                logging.info('Row contains no tokens.')
            _data += tokens
            t.update(1)
    return torch.Tensor(_data).long()


class LanguageModelingDataset(torch.utils.data.Dataset):
    """Defines a dataset for language modeling."""

    def __init__(self, data, vocab):
        """Create a LanguageModelingDataset given a path and a field.

        Arguments:
            path: Path to the data file.
        """
        super(LanguageModelingDataset, self).__init__()
        self._data = data
        self._vocab = vocab

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_vocab(self):
        return self._vocab


def _setup_datasets(dataset_name, tokenizer=get_tokenizer("basic_english"),
                    root='.data', vocab=None, include_unk=False):
    if dataset_name == 'PennTreebank':
        train_path = download_from_url(URLS['PennTreebank'][0], root=root)
        test_path = download_from_url(URLS['PennTreebank'][1], root=root)
        valid_path = download_from_url(URLS['PennTreebank'][2], root=root)
    else:
        dataset_tar = download_from_url(URLS[dataset_name], root=root)
        extracted_files = extract_archive(dataset_tar)
        for fname in extracted_files:
            if 'train' in fname:
                train_path = os.path.join(root, fname)
            if 'test' in fname:
                test_path = os.path.join(root, fname)
            if 'valid' in fname:
                valid_path = os.path.join(root, fname)

    if vocab is None:
        logging.info('Building Vocab based on {}'.format(train_path))
        vocab = build_vocab_from_iterator(_read_text_iterator(train_path, tokenizer))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info('Vocab has {} entries'.format(len(vocab)))
    logging.info('Creating training data')
    train_data = _create_data_from_iterator(
        vocab, _read_text_iterator(train_path, tokenizer), include_unk)
    logging.info('Creating testing data')
    test_data = _create_data_from_iterator(
        vocab, _read_text_iterator(test_path, tokenizer), include_unk)
    logging.info('Creating valid data')
    valid_data = _create_data_from_iterator(
        vocab, _read_text_iterator(valid_path, tokenizer), include_unk)
    return (LanguageModelingDataset(train_data, vocab),
            LanguageModelingDataset(test_data, vocab),
            LanguageModelingDataset(valid_data, vocab))


def WikiText2(*args, **kwargs):
    return _setup_datasets(*(("WikiText2",) + args), **kwargs)


def WikiText103(*args, **kwargs):
    return _setup_datasets(*(("WikiText103",) + args), **kwargs)


def PennTreebank(*args, **kwargs):
    return _setup_datasets(*(("PennTreebank",) + args), **kwargs)
