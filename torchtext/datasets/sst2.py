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
    IterableAsDataPipe,
)


from torch.utils.data.datapipes.iter import LoadFilesFromDisk
NUM_LINES = {}
MD5 = {}
URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"

DATASET_NAME = "SST2"
_PATH = "SST-2.zip"


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'dev', 'test'))
def SST2(root, split):
    # TODO: not working: cache_dp = IterableAsDataPipe([URL]).on_disk_cache(HttpReader, filepath_fn=lambda x: os.path.join(root, os.path.basename(x)))
    saver_dp = HttpReader([URL]).map(lambda x: (x[0],x[1].read())).save_to_disk(filepath_fn=lambda x: os.path.join(root, os.path.basename(x)))
    extracted_files=LoadFilesFromDisk(saver_dp).read_from_zip()
    return extracted_files.filter(lambda x: split in x[0]).parse_csv_files(skip_header = True, delimiter='\t').map(lambda x: (x[1], int(x[2])))
