from torchtext.utils import download_from_url
import json
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import check_default_set

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


def _setup_datasets(dataset_name, root='.data', data_select=('train', 'dev')):
    data_select = check_default_set(data_select, ('train', 'dev'))
    extracted_files = []
    select_to_index = {'train': 0, 'dev': 1}
    extracted_files = [download_from_url(URLS[dataset_name][select_to_index[key]],
                                         root=root) for key in select_to_index.keys()]
    return tuple(RawTextIterableDataset(dataset_name, NUM_LINES[dataset_name], _create_data_from_json(extracted_files[select_to_index[item]])) for item in data_select)


def SQuAD1(*args, **kwargs):
    """ A dataset iterator yields the data of Stanford Question Answering dataset - SQuAD1.0.
    The iterator yields a tuple of (raw context, raw question, a list of raw answer, a list of answer positions in the raw context).
    For example, ('Architecturally, the school has a Catholic character. Atop the ...',
                  'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
                  ['Saint Bernadette Soubirous'],
                  [515])

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        data_select: a string or tuple for the returned datasets (Default: ('train', 'dev'))
            By default, both datasets (train, dev) are generated. Users could also choose any one or two of them,
            for example ('train', 'dev') or just a string 'train'.

    Examples:
        >>> train_dataset, dev_dataset = torchtext.experimental.datasets.raw.SQuAD1()
        >>> for idx, (context, question, answer, ans_pos) in enumerate(train_dataset):
        >>>     print(idx, (context, question, answer, ans_pos))
    """

    return _setup_datasets(*(("SQuAD1",) + args), **kwargs)


def SQuAD2(*args, **kwargs):
    """ A dataset iterator yields the data of Stanford Question Answering dataset - SQuAD2.0.
    The iterator yields a tuple of (raw context, raw question, a list of raw answer, a list of answer positions in the raw context).
    For example, ('Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an ...',
                  'When did Beyonce start becoming popular?',
                  ['in the late 1990s'],
                  [269])

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        data_select: a string or tuple for the returned datasets (Default: ('train', 'dev'))
            By default, both datasets (train, dev) are generated. Users could also choose any one or two of them,
            for example ('train', 'dev') or just a string 'train'.

    Examples:
        >>> train_dataset, dev_dataset = torchtext.experimental.datasets.raw.SQuAD2()
        >>> for idx, (context, question, answer, ans_pos) in enumerate(train_dataset):
        >>>     print(idx, (context, question, answer, ans_pos))
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
