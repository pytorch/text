from torchtext.datasets.ag_news import DATASET_NAME, NUM_LINES
from torchtext.data.datasets_utils import (
    _create_dataset_directory,
    _wrap_split_argument,
)
import os
from datapipes.iter import(
    CSVParser,
    ReadFilesFromZip,
    HttpReader,
)


NUM_LINES = {}
MD5 = {}
URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"

DATASET_NAME = "SST2"
_PATH = "SST-2.zip"


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'dev', 'test'))
def SST2(root, split):
    save_path = os.path.join(root, _PATH)
    http_list = [d for d in HttpReader([URL]).map(lambda x: x[1])]
    with open(save_path, 'wb') as f:
        f.write(http_list[0].read())
    extracted_files = ReadFilesFromZip([(os.path.dirname(save_path), open(save_path, 'rb'))])
    return extracted_files.filter(lambda x: split in x[0]).parse_csv_files(delimiter='\t').map(lambda x: (x[1],x[2]))
