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
    """Defines a dataset for language modeling.
       Currently, we only support the following datasets:

             - WikiText2
             - WikiText103
             - PennTreebank

    """

    def __init__(self, data, vocab):
        """Initiate language modeling dataset.

        Arguments:
            data: a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.Tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab: Vocabulary object used for dataset.

        Examples:
            >>> from torchtext.vocab import build_vocab_from_iterator
            >>> data = torch.Tensor([token_id_1, token_id_2,
                                     token_id_3, token_id_1]).long()
            >>> vocab = build_vocab_from_iterator([['language', 'modeling']])
            >>> dataset = LanguageModelingDataset(data, vocab)

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
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> from torchtext.datasets import WikiText2
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = WikiText2(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()

    """

    return _setup_datasets(*(("WikiText2",) + args), **kwargs)


def WikiText103(*args, **kwargs):
    """ Defines WikiText103 datasets.

    Create language modeling dataset: WikiText103
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> from torchtext.datasets import WikiText103
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = WikiText103(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()

    """

    return _setup_datasets(*(("WikiText103",) + args), **kwargs)


def PennTreebank(*args, **kwargs):
    """ Defines PennTreebank datasets.

    Create language modeling dataset: PennTreebank
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> from torchtext.datasets import PennTreebank
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = PennTreebank(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()

    """

    return _setup_datasets(*(("PennTreebank",) + args), **kwargs)
