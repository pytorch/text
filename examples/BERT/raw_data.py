import torch
import io
from torchtext.utils import download_from_url, extract_archive

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


class RawLanguageModelingDataset(torch.utils.data.IterableDataset):
    """Defines an abstraction for raw language modeling iterable datasets.
    """

    def __init__(self, iterator):
        """Initiate language modeling dataset.
        """
        super(RawLanguageModelingDataset, self).__init__()
        self._iterator = iterator
        self.has_setup = False
        self.start = 0
        self.num_lines = None

    def setup_iter(self, start=0, num_lines=None):
        self.start = start
        self.num_lines = num_lines
        self.has_setup = True

    def __iter__(self):
        if not self.has_setup:
            self.setup_iter()

        for i, item in enumerate(self._iterator):
            if i >= self.start:
                yield item
            if self.num_lines is not None and i == (self.start + self.num_lines):
                break

    def get_iterator(self):
        return self._iterator


def _create_data(data_path):
    with io.open(data_path, encoding="utf8") as f:
        for line in f:
            if line != "":
                yield line


def _setup_datasets(dataset_name, root='.data'):

    if dataset_name == 'PennTreebank':
        extracted_files = []
        select_to_index = {'train': 0, 'test': 1, 'valid': 2}
        extracted_files = [download_from_url(URLS['PennTreebank'][select_to_index[key]],
                                             root=root) for key in data_select]
    else:
        dataset_tar = download_from_url(URLS[dataset_name], root=root)
        extracted_files = extract_archive(dataset_tar)

    for fname in extracted_files:
        if 'train' in fname:
            train_path = fname
        if 'valid' in fname:
            valid_path = fname
        if 'test' in fname:
            test_path = fname
    return (RawLanguageModelingDataset(_create_data(train_path)),
            RawLanguageModelingDataset(_create_data(valid_path)),
            RawLanguageModelingDataset(_create_data(test_path)))


def WikiText103(*args, **kwargs):
    return _setup_datasets(*(("WikiText103",) + args), **kwargs)
