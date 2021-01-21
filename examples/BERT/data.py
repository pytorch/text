import glob
import torch
import logging
from torchtext.data.utils import get_tokenizer
import random
from torchtext.experimental.datasets import LanguageModelingDataset
from torch.utils.data.datasets import ListDirFilesIterableDataset, LoadFilesFromDiskIterableDataset


###################################################################
# Set up dataset for book corpus
###################################################################
def BookCorpus(vocab, tokenizer=get_tokenizer("basic_english"),
               data_select=('train', 'valid', 'test'), removed_tokens=[],
               min_sentence_len=None):

    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(('train', 'test', 'valid'))):
        raise TypeError('data_select is not supported!')

    extracted_files = glob.glob('/datasets01/bookcorpus/021819/*/*.txt')
    random.seed(1000)
    random.shuffle(extracted_files)

    num_files = len(extracted_files)
    _path = {'train': extracted_files[:(num_files // 20 * 17)],
             'test': extracted_files[(num_files // 20 * 17):(num_files // 20 * 18)],
             'valid': extracted_files[(num_files // 20 * 18):]}

    data = {}
    for item in _path.keys():
        data[item] = []
        logging.info('Creating {} data'.format(item))
        tokens = []
        for txt_file in _path[item]:
            with open(txt_file, 'r', encoding="utf8", errors='ignore') as f:
                for line in f.readlines():
                    _tokens = tokenizer(line.strip())
                    if min_sentence_len:
                        if len(_tokens) >= min_sentence_len:
                            tokens.append([vocab.stoi[token] for token in _tokens])
                    else:
                        tokens += [vocab.stoi[token] for token in _tokens]
        data[item] = tokens

    for key in data_select:
        if data[key] == []:
            raise TypeError('Dataset {} is empty!'.format(key))
    if min_sentence_len:
        return tuple(LanguageModelingDataset(data[d], vocab, lambda x: x, False)
                     for d in data_select)
    else:
        return tuple(LanguageModelingDataset(torch.tensor(data[d]).long(), vocab, lambda x: x, False)
                     for d in data_select)


class CC100(torch.utils.data.IterableDataset):
    def __init__(self, data_directory, languages, start_line=0, chunk=16):
        """

        Examples:
            >>> from data import CC100
            >>> dataset = CC100('/datasets01/cc100/031720/', {'zh_TW.txt', 'ja_XX.txt'}, start_line=30, chunk=10)
            >>> for rec in dataset:
            >>>     print(rec)
        """

        file_paths = ListDirFilesIterableDataset(data_directory, languages)
        self.dataset_list = [item[1] for item in LoadFilesFromDiskIterableDataset(file_paths)]
        self.start_line = start_line
        self.chunk = chunk
        self._count = 0
        self._current_dataset = 0

    def __iter__(self):
        for i, dataset_handle in enumerate(self.dataset_list):
            self.setup_dataset(dataset_handle)
            for _count in range(self.chunk):
                _text = self.readline(dataset_handle)
                yield _text.decode('utf-8')

    def setup_dataset(self, dataset_handle):
        for _count in range(self.start_line):
            _text = self.readline(dataset_handle)

    def readline(self, dataset_handle):
        _text = dataset_handle.readline()
        while _text == b'\n':
            _text = dataset_handle.readline()
        return _text
