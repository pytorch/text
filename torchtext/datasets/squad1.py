from torchtext.utils import download_from_url
import json
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header

URL = {
    'train': "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    'dev': "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}

MD5 = {
    'train': "981b29407e0affa3b1b156f72073b945",
    'dev': "3e85deb501d4e538b6bc56f786231552",
}

NUM_LINES = {
    'train': 87599,
    'dev': 10570,
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


@_add_docstring_header(num_lines=NUM_LINES)
@_wrap_split_argument(('train', 'dev'))
def SQuAD1(root, split):
    extracted_files = download_from_url(URL[split], root=root, hash_value=MD5[split], hash_type='md5')
    return _RawTextIterableDataset('SQuAD1', NUM_LINES[split],
                                   _create_data_from_json(extracted_files))
