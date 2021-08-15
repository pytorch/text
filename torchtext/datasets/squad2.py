from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
)
import os
from torchtext.data.data_pipes import ParseSQuADQAData
from datapipes.iter import (
    HttpReader,
    IterableAsDataPipe,
)

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


DATASET_NAME = "SQuAD2"


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'dev'))
def SQuAD2(root, split):
    """Demonstrates use case when more complex processing is needed on data-stream
        Here we process dictionary returned by standard JSON reader and write custom datapipe to orchestrates data samples for Q&A use-case
    """
    
    # cache data on-disk
    cache_dp = IterableAsDataPipe([URL]).on_disk_cache(HttpReader, op_map=lambda x: (x[0], x[1].read()), filepath_fn=lambda x: os.path.join(root, os.path.basename(x)))

    # do sanity check
    check_cache_dp = cache_dp.check_hash({os.path.join(root, os.path.basename(URL[split])): MD5}, 'md5')

    # stack custom data pipe on top of JSON reader to orchestrate data samples for Q&A dataset
    return ParseSQuADQAData(check_cache_dp.parse_json_files())
