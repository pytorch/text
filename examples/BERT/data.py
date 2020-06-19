import glob
import torch
import logging
from torchtext.data.utils import get_tokenizer
import random


class LanguageModelingDataset(torch.utils.data.Dataset):
    """Defines a dataset for language modeling.
    """

    def __init__(self, data, vocab):
        """Initiate language modeling dataset.
        """

        super(LanguageModelingDataset, self).__init__()
        self.data = data
        self.vocab = vocab

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x

    def get_vocab(self):
        return self.vocab


###################################################################
# Set up dataset for book corpus
###################################################################
def BookCorpus(vocab, tokenizer=get_tokenizer("basic_english"),
               data_select=('train', 'test', 'valid'), removed_tokens=[],
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
        return tuple(LanguageModelingDataset(data[d], vocab)
                     for d in data_select)
    else:
        return tuple(LanguageModelingDataset(torch.tensor(data[d]).long(), vocab)
                     for d in data_select)
