import torch
import json
import logging
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
import io
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.functional import numericalize_tokens_from_iterator
from torchtext.data.functional import custom_replace
import random
import glob
import os
import raw_data

URLS = {
    'SQuAD1':
        ['https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
         'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'],
    'SQuAD2':
        ['https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json',
         'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json'],
    'WikiText2':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
    'WikiText103':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip',
    'PennTreebank':
        ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt'],
    'WMTNewsCrawl': 'http://www.statmt.org/wmt11/training-monolingual-news-2010.tgz'
}

DATASETS = {
    'WikiText103': raw_data.WikiText103,
}


class QuestionAnswerDataset(torch.utils.data.Dataset):
    """Defines a dataset for question answer.
       Currently, we only support the followVing datasets:
             - SQuAD1.1
    """

    def __init__(self, data, vocab):
        super(QuestionAnswerDataset, self).__init__()
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


def create_data_from_iterator(vocab, processed_data, tokenizer):
    for items in processed_data:
        _ans = []
        for idx in range(len(items['answer_start'])):
            ans_start_idx = items['answer_start'][idx]
            if ans_start_idx == -1:  # No answer for this sample
                _ans.append((iter(vocab[token] for token in tokenizer(items['answers'][idx])),
                             ans_start_idx, ans_start_idx))
            else:
                ans_start_token_id = len(tokenizer(items['context'][:ans_start_idx]))
                ans_end_token_id = ans_start_token_id + len(tokenizer(items['answers'][idx])) - 1
                _ans.append((iter(vocab[token] for token in tokenizer(items['answers'][idx])),
                             ans_start_token_id, ans_end_token_id))
        yield iter(vocab[token] for token in tokenizer(items['context'])), \
            iter(vocab[token] for token in tokenizer(items['question'])), _ans


def squad_iterator(processed_data, tokenizer):
    for items in processed_data:
        yield tokenizer(items['context']) \
            + tokenizer(items['question']) \
            + tokenizer(items['answers'])


def process_raw_json_data(raw_json_data):
    processed = []
    for layer1 in raw_json_data:
        for layer2 in layer1['paragraphs']:
            for layer3 in layer2['qas']:
                processed.append({'context': layer2['context'],
                                  'question': layer3['question'],
                                  'answers': [item['text'] for item in layer3['answers']],
                                  'answer_start': [item['answer_start'] for item in layer3['answers']]})
                if len(processed[-1]['answers']) == 0:
                    processed[-1]['answers'] = [""]
                    processed[-1]['answer_start'] = [-1]
    return processed


def _setup_qa_datasets(dataset_name, tokenizer=get_tokenizer("basic_english"),
                       root='.data', vocab=None, removed_tokens=[],
                       data_select=('train', 'dev')):

    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(('train', 'dev'))):
        raise TypeError('data_select is not supported!')

    extracted_files = []
    select_to_index = {'train': 0, 'dev': 1}
    extracted_files = [download_from_url(URLS[dataset_name][select_to_index[key]],
                                         root=root) for key in data_select]

    squad_data = {}
    for item in data_select:
        with open(extracted_files[select_to_index[item]]) as json_file:
            raw_data = json.load(json_file)['data']
            squad_data[item] = process_raw_json_data(raw_data)

    if vocab is None:
        if 'train' not in squad_data.keys():
            raise TypeError("Must pass a vocab if train is not selected.")
        logging.info('Building Vocab based on train data')
        vocab = build_vocab_from_iterator(squad_iterator(squad_data['train'], tokenizer))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info('Vocab has {} entries'.format(len(vocab)))

    data = {}
    for item in data_select:
        data_iter = create_data_from_iterator(vocab, squad_data[item], tokenizer)
        tensor_data = []
        for context, question, _ans in data_iter:
            iter_data = {'context':
                         torch.tensor([token_id for token_id in context]).long(),
                         'question':
                         torch.tensor([token_id for token_id in question]).long(),
                         'answers': [],
                         'ans_pos': []}
            for (_answer, ans_start_id, ans_end_id) in _ans:
                iter_data['answers'].append(torch.tensor([token_id for token_id in _answer]).long())
                iter_data['ans_pos'].append(torch.tensor([ans_start_id, ans_end_id]).long())
            tensor_data.append(iter_data)
        data[item] = tensor_data

    return tuple(QuestionAnswerDataset(data[item], vocab) for item in data_select)


def SQuAD1(*args, **kwargs):
    return _setup_qa_datasets(*(('SQuAD1',) + args), **kwargs)


def SQuAD2(*args, **kwargs):
    return _setup_qa_datasets(*(('SQuAD2',) + args), **kwargs)


###################################################################
# Set up WMTNewsCrawl
# Need a larger dataset to train BERT model
###################################################################
class TextDataset(torch.utils.data.Dataset):
    """Defines an abstract text datasets.
    """

    def __init__(self, data, vocab, transforms):
        """Initiate text dataset.
        Arguments:
            data: a list of text string tuples.
            vocab: Vocabulary object used for dataset.
            transforms: a tuple of text string transforms.
        """

        super(TextDataset, self).__init__()
        self.data = data
        self.vocab = vocab
        self.transforms = transforms

    def __getitem__(self, i):
        print(len(self.data[i]))
        return tuple(self.transforms[j](self.data[i][j]) for j in range(len(self.data[i])))

    def __len__(self):
        return len(self.data)

    def get_vocab(self):
        return self.vocab


class LanguageModelingDataset(torch.utils.data.Dataset):
    """Defines a dataset for language modeling.
       Currently, we only support the following datasets:
             - WikiText2
             - WikiText103
             - PennTreebank
             - WMTNewsCrawl
    """

    def __init__(self, data, vocab):
        """Initiate language modeling dataset.
        Arguments:
            data: a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab: Vocabulary object used for dataset.
        Examples:
            >>> from torchtext.vocab import build_vocab_from_iterator
            >>> data = torch.tensor([token_id_1, token_id_2,
                                     token_id_3, token_id_1]).long()
            >>> vocab = build_vocab_from_iterator([['language', 'modeling']])
            >>> dataset = LanguageModelingDataset(data, vocab)
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


def _get_datafile_path(key, extracted_files):
    for fname in extracted_files:
        if key in fname:
            return fname


def _setup_datasets(dataset_name, tokenizer=get_tokenizer("basic_english"),
                    root='.data', vocab=None, removed_tokens=[],
                    data_select=('train', 'test', 'valid'), **kwargs):

    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(('train', 'test', 'valid'))):
        raise TypeError('data_select is not supported!')

    if dataset_name == 'PennTreebank':
        extracted_files = []
        select_to_index = {'train': 0, 'test': 1, 'valid': 2}
        extracted_files = [download_from_url(URLS['PennTreebank'][select_to_index[key]],
                                             root=root) for key in data_select]
    elif dataset_name == 'WMTNewsCrawl':
        if not (data_select == ['train'] or set(data_select).issubset(set(('train',)))):
            raise ValueError("WMTNewsCrawl only creates a training dataset. "
                             "data_select should be 'train' "
                             "or ('train',), got {}.".format(data_select))
        dataset_tar = download_from_url(URLS[dataset_name], root=root)
        extracted_files = extract_archive(dataset_tar)
        year = kwargs.get('year', 2010)
        language = kwargs.get('language', 'en')
        fname = 'news.{}.{}.shuffled'.format(year, language)
        extracted_files = [f for f in extracted_files if fname in f]
    else:
        dataset_tar = download_from_url(URLS[dataset_name], root=root)
        extracted_files = extract_archive(dataset_tar)

    _path = {}
    for item in data_select:
        _path[item] = _get_datafile_path(item, extracted_files)

    if vocab is None:
        if 'train' not in _path.keys():
            raise TypeError("Must pass a vocab if train is not selected.")
        logging.info('Building Vocab based on {}'.format(_path['train']))
        txt_iter = iter(tokenizer(row) for row in io.open(_path['train'],
                                                          encoding="utf8"))
        vocab = build_vocab_from_iterator(txt_iter)
        logging.info('Vocab has {} entries'.format(len(vocab)))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")

    data = {}
    for item in _path.keys():
        data[item] = []
        logging.info('Creating {} data'.format(item))
        txt_iter = iter(tokenizer(row) for row in io.open(_path[item],
                                                          encoding="utf8"))
        _iter = numericalize_tokens_from_iterator(
            vocab, txt_iter, removed_tokens)
        for tokens in _iter:
            data[item] += [token_id for token_id in tokens]

    for key in data_select:
        if data[key] == []:
            raise TypeError('Dataset {} is empty!'.format(key))

    return tuple(LanguageModelingDataset(torch.tensor(data[d]).long(), vocab)
                 for d in data_select)


def WMTNewsCrawl(*args, **kwargs):
    """ Defines WMT News Crawl.
        Examples:
            >>> from torchtext.experimental.datasets import WMTNewsCrawl
            >>> from torchtext.data.utils import get_tokenizer
            >>> tokenizer = get_tokenizer("spacy")
            >>> train_dataset, = WMTNewsCrawl(tokenizer=tokenizer,
                                              data_select='train',
                                              language='en')
            >>> vocab = train_dataset.get_vocab()
        """
    return _setup_datasets(*(("WMTNewsCrawl",) + args), **kwargs)


###################################################################
# Set up dataset for  Next Sentence Prediction
###################################################################


def vocab_func(vocab):
    def _forward(tok_iter):
        return [vocab[tok] for tok in tok_iter]
    return _forward


def totensor(dtype):
    def _forward(ids_list):
        return torch.tensor(ids_list).to(dtype)
    return _forward


def build_vocab(data, transforms):
    tok_list = []
    for _, txt in data:
        tok_list.append(transforms(txt))
    return build_vocab_from_iterator(tok_list)


def squential_transforms(*transforms):
    def _forward(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return _forward


def _setup_ns(dataset_name, tokenizer=get_tokenizer("basic_english"),
              root='.data', vocab=None, removed_tokens=[],
              data_select=('train', 'test', 'valid'), single_line=True):

    text_transform = []
    if tokenizer is None:
        tokenizer = get_tokenizer('basic_english')
    text_transform = squential_transforms(tokenizer)

    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(('train', 'test', 'valid'))):
        raise TypeError('Given data selection {} is not supported!'.format(data_select))

    train, valid, test = DATASETS[dataset_name](root=root)
    # Cache raw text iterable dataset
    raw_data = {'train': [(txt,) for txt in train],
                'valid': [(txt,) for txt in valid],
                'test': [(txt,) for txt in test]}
    if single_line:
        for item in raw_data.keys():
            raw_data[item] = ' '.join(raw_data[item])

    if vocab is None:
        if 'train' not in data_select:
            raise TypeError("Must pass a vocab if train is not selected.")
        tok_list = []
        for txt in raw_data['train']:
            tok_list.append(text_transform(txt))
        vocab = build_vocab_from_iterator(tok_list)
    text_transform = squential_transforms(text_transform, vocab_func(vocab),
                                          totensor(dtype=torch.long))
    return tuple(TextDataset(raw_data[item], vocab, (text_transform,)) for item in data_select)


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
        data_select: the returned datasets (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test').
            If 'train' is not in the tuple, an vocab object should be provided which will
            be used to process valid and/or test data.
        removed_tokens: removed tokens from output dataset (Default: [])
        data_select: a string or tupel for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.
        single_line: collapse the multiple text into single line (Default: True)

    Examples:
        >>> from torchtext.experimental.datasets import WikiText103
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = WikiText103(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
        >>> valid_dataset, = WikiText103(tokenizer=tokenizer, vocab=vocab,
                                         data_select='valid')
    """

    return _setup_ns(*(("WikiText103",) + args), **kwargs)


###################################################################
# Set up dataset for  Next Sentence Prediction
###################################################################


_patterns = [(r'<.*>', ''),
             (r'&amp;', '&'),
             (r'&lt;', '<'),
             (r'&gt;', '>'),
             (r'<ref[^<]*<\/ref>', ''),
             (r'<[^>]*>', ''),
             (r'\[http:[^] ]*', '['),
             (r'\|thumb', ''),
             (r'\|left', ''),
             (r'\|right', ''),
             (r'\|\d+px', ''),
             (r'\[\[image:[^\[\]]*\|', ''),
             (r'\[\[category:([^|\]]*)[^]]*\]\]', '[[$1]]'),
             (r'\[\[[a-z\-]*:[^\]]*\]\]', ''),
             (r'\[\[[^\|\]]*\|', '[['),
             (r'\{\{[^\}]*\}\}', ''),
             (r'\{[^\}]*\}', ''),
             (r'\[', ''),
             (r'\]', ''),
             (r'&[^;]*;', ' '),
             (r'A', 'a'), (r'B', 'b'), (r'C', 'c'),
             (r'D', 'd'), (r'E', 'e'), (r'F', 'f'),
             (r'G', 'g'), (r'H', 'h'), (r'I', 'i'),
             (r'J', 'j'), (r'K', 'k'), (r'L', 'l'),
             (r'M', 'm'), (r'N', 'n'), (r'O', 'o'),
             (r'P', 'p'), (r'Q', 'q'), (r'R', 'r'),
             (r'S', 's'), (r'T', 't'), (r'U', 'u'),
             (r'V', 'v'), (r'W', 'w'), (r'X', 'x'),
             (r'Y', 'y'), (r'Z', 'z'),
             (r'0', ' zero '), (r'1', ' one '), (r'2', ' two '),
             (r'3', ' three '), (r'4', ' four '), (r'5', ' five '),
             (r'6', ' six '), (r'7', ' seven '), (r'8', ' eight '),
             (r'9', ' nine '),
             (r'[^a-z\n]+', ' '),
             (r'\n ', ''),
             (r'\s+', ' '),
             (r'\n\s*\n', r'\n')
             ]
enwik9_norm_transform = custom_replace(_patterns)


def generate_offsets(filename):
    offsets = []
    with open(filename) as f:
        offsets.append(f.tell())
        while f.readline():
            offsets.append(f.tell())
    return offsets


def read_lines_from_iterator(data_path, offsets, begin_line, num_lines):
    with open(data_path) as f:
        f.seek(offsets[begin_line])
        for i in range(num_lines):
            yield f.tell(), f.readline()


def preprocess_raw_enwik9(input_filename, output_filename):
    with open(input_filename, 'r') as f1:
        with open(output_filename, 'w') as f2:
            while True:
                line = f1.readline()
                if not line:
                    break
                line = list(enwik9_norm_transform([line]))[0]
                if line != ' ' and line != '':
                    if line[0] == ' ':
                        line = line[1:]
                    f2.writelines(line + '\n')


class EnWik9(torch.utils.data.Dataset):
    r"""Compressed size of first 10^9 bytes of enwiki-20060303-pages-articles.xml.
        It's part of Large Text Compression Benchmark project
    """

    def __init__(self, begin_line=0, num_lines=6348957, root='.data'):
        """Initiate EnWik9 dataset.
        Arguments:
            begin_line: the number of beginning line. Default: 0
            num_lines: the number of lines to be loaded. Default: 6348957
            root: Directory where the datasets are saved. Default: ".data"
            data: a list of label/tokens tuple. tokens are a tensor after
        """

        super(EnWik9, self).__init__()

        processed_file = os.path.join(root, 'norm_enwik9')
        if not os.path.exists(processed_file):
            url = 'http://mattmahoney.net/dc/enwik9.zip'
            dataset_zip = download_from_url(url,
                                            path=os.path.join(root, 'enwik9.zip'),
                                            root=root)
            extracted_file = extract_archive(dataset_zip)
            raw_file = extracted_file[0]
            preprocess_raw_enwik9(raw_file, processed_file)

        # Meta information
        offsets = generate_offsets(processed_file)
        read_lines = read_lines_from_iterator(processed_file,
                                              offsets, begin_line, num_lines)

        self.file_f = open(processed_file)
        self.start_offsets = []
        self.end_offsets = []
        for (line_num, line_str) in read_lines:
            _idx = line_num
            for item in line_str.split():
                self.start_offsets.append(_idx)
                self.end_offsets.append(_idx + len(item))
                _idx += len(item) + 1

        self._vocab = None

    def __getitem__(self, index):
        def _get_item(i):
            start_idx = self.start_offsets[i]
            end_idx = self.end_offsets[i]
            self.file_f.seek(start_idx)
            return self.file_f.read(end_idx - start_idx)
        if isinstance(index, slice):
            return [_get_item(ii) for ii in range(*index.indices(len(self)))]
        elif isinstance(index, int):
            return _get_item(index)

    def __len__(self):
        return len(self.start_offsets)


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
