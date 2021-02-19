import json
from torchtext.utils import download_from_url, extract_archive
from torchtext.datasets.common import RawTextIterableDataset
from torchtext.datasets.common import wrap_split_argument
from torchtext.datasets.common import add_docstring_header
from torchtext.datasets.common import find_match

URL = 'http://nlp.stanford.edu/projects/snli/snli_1.0.zip'

MD5 = '981c3df556bbaea3f17f752456d0088c'

NUM_LINES = {
    'train': 550152,
    'dev': 10000,
    'test': 10000,
}


def _create_data_from_jsonlines(data_path):
    with open(data_path) as jsonlines:
        for content in jsonlines:
            json_content = json.loads(content)
            yield (json_content['annotator_labels'], json_content['gold_label'],
                   json_content['sentence1'], json_content['sentence1_binary_parse'], json_content['sentence1_parse'],
                   json_content['sentence2'], json_content['sentence2_binary_parse'], json_content['sentence2_parse'])


@wrap_split_argument
@add_docstring_header()
def SNLI(root='.data', split=('train', 'dev', 'test'), offset=0):
    dataset_tar = download_from_url(URL, root=root, hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)
    datasets = []
    for item in split:
        path = find_match(item + '.jsonl', extracted_files)
        datasets.append(RawTextIterableDataset("SNLI", NUM_LINES[item],
                                               _create_data_from_jsonlines(path), offset=offset))
    return datasets
