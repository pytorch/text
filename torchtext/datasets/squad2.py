from torchtext.utils import download_from_url
import json
from torchtext.data.datasets_utils import RawTextIterableDataset
from torchtext.data.datasets_utils import wrap_split_argument
from torchtext.data.datasets_utils import add_docstring_header

URL = {
    'train': "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
    'dev': "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
}

MD5 = {
    'train': "62108c273c268d70893182d5cf8df740",
    'dev': "246adae8b7002f8679c027697b0b7cf8",
}

NUM_LINES = {
    'train': 130319,
    'dev': 11873,
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


@wrap_split_argument
@add_docstring_header()
def SQuAD2(root='.data', split=('train', 'dev')):
    datasets = []
    for item in split:
        extracted_files = download_from_url(URL[item], root=root, hash_value=MD5[item], hash_type='md5')
        datasets.append(RawTextIterableDataset('SQuAD2', NUM_LINES[item],
                                               _create_data_from_json(extracted_files)))
    return datasets
