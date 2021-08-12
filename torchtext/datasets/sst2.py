from torchtext.data.datasets_utils import (
    _create_dataset_directory,
    _wrap_split_argument,
)
import os
from datapipes.iter import (
    HttpReader,
    IterableAsDataPipe,
)


NUM_LINES = {
    'train': 6739,
    'dev': 872,
    'test': 1821,
}

MD5 = "9f81648d4199384278b86e315dac217c"
URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"

_EXTRACTED_FILES = {
    'train': f'{os.sep}'.join(['SST-2', 'train.tsv']),
    'dev': f'{os.sep}'.join(['SST-2', 'dev.tsv']),
    'test': f'{os.sep}'.join(['SST-2', 'test.tsv']),
}

_EXTRACTED_FILES_MD5 = {
    'train': 'da409a0a939379ed32a470bc0f7fe99a',
    'dev': '268856b487b2a31a28c0a93daaff7288',
    'test': '3230e4efec76488b87877a56ae49675a',
}

DATASET_NAME = "SST2"


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'dev', 'test'))
def SST2(root, split):
    cache_dp = IterableAsDataPipe([URL]).on_disk_cache(HttpReader, op_map=lambda x: (x[0], x[1].read()), filepath_fn=lambda x: os.path.join(root, os.path.basename(x)))
    check_cache_dp = cache_dp.check_hash({os.path.join(root, "SST-2.zip"): MD5}, 'md5')
    extracted_files = check_cache_dp.read_from_zip()
    check_extracted_files = extracted_files.filter(lambda x: split in x[0]).check_hash({os.path.join(root, _EXTRACTED_FILES[split]): _EXTRACTED_FILES_MD5[split]}, 'md5')
    return check_extracted_files.parse_csv_files(skip_header=True, delimiter='\t').map(lambda x: (x[1], x[2]))
