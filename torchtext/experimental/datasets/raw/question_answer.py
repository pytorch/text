import torch
from torchtext.utils import download_from_url
import json
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset

URLS = {
    'SQuAD1':
        ['https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
         'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'],
    'SQuAD2':
        ['https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json',
         'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json']
}


def _create_data_from_json(data_path):
    with open(data_path) as json_file:
        raw_json_data = json.load(json_file)['data']
        for layer1 in raw_json_data:
            for layer2 in layer1['paragraphs']:
                for layer3 in layer2['qas']:
                    _context, _question = layer2['context'], layer3['question']
                    _answers = [item['text'] for item in layer3['answers']]
                    _answer_start = [item['answer_start'] for item in layer3['answers']]
                    if len(_answers) == 0:
                        _answers = [""]
                        _answer_start = [-1]
                    # yield the raw data in the order of context, question, answers, answer_start
                    yield (_context, _question, _answers, _answer_start)


def _setup_datasets(dataset_name, root='.data'):
    extracted_files = []
    select_to_index = {'train': 0, 'dev': 1}
    extracted_files = [download_from_url(URLS[dataset_name][select_to_index[key]],
                                         root=root) for key in select_to_index.keys()]
    train_iter = _create_data_from_json(extracted_files[0])
    dev_iter = _create_data_from_json(extracted_files[1])
    return (RawTextIterableDataset(dataset_name, NUM_LINES[dataset_name], train_iter),
            RawTextIterableDataset(dataset_name, NUM_LINES[dataset_name], dev_iter))


def SQuAD1(*args, **kwargs):
    """ Defines SQuAD1 datasets.

    Examples:
        >>> train, dev = torchtext.experimental.datasets.raw.SQuAD1()
    """

    return _setup_datasets(*(("SQuAD1",) + args), **kwargs)


def SQuAD2(*args, **kwargs):
    """ Defines SQuAD2 datasets.

    Examples:
        >>> train, dev = torchtext.experimental.datasets.raw.SQuAD2()
    """

    return _setup_datasets(*(("SQuAD2",) + args), **kwargs)


DATASETS = {
    'SQuAD1': SQuAD1,
    'SQuAD2': SQuAD2
}
NUM_LINES = {
    'SQuAD1': 87599,
    'SQuAD2': 130319
}
