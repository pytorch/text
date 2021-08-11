from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
)
import os
from torchtext.data.data_pipes import JSONParserIterDataPipe
from datapipes.iter import (
    HttpReader,
)
from torch.utils.data.datapipes.iter import LoadFilesFromDisk

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


DATASET_NAME = "SQuAD1"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'dev'))
def SQuAD1(root, split):
    saver_dp = HttpReader([URL[split]]).map(lambda x: (x[0], x[1].read())).save_to_disk(filepath_fn=lambda x: os.path.join(root, os.path.basename(x)))
    return LoadFilesFromDisk(saver_dp).parse_json_files()
