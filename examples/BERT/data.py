import torch
import json
import logging
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

URLS = {
    'SQuAD1':
        ['https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
         'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'],
    'SQuAD2':
        ['https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json',
         'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json']
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
