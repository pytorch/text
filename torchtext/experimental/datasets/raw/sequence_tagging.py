from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
<<<<<<< HEAD
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
=======
from torchtext.experimental.datasets.raw.common import check_default_set
>>>>>>> upstream/fbsync

URLS = {
    "UDPOS":
    'https://bitbucket.org/sivareddyg/public/downloads/en-ud-v2.zip',
    "CoNLL2000Chunking": {
        'train': 'https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz',
        'test': 'https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz'
    }
}


def _create_data_from_iob(data_path, separator="\t"):
    with open(data_path, encoding="utf-8") as input_file:
        columns = []
        for line in input_file:
            line = line.strip()
            if line == "":
                if columns:
                    yield columns
                columns = []
            else:
                for i, column in enumerate(line.split(separator)):
                    if len(columns) < i + 1:
                        columns.append([])
                    columns[i].append(column)
        if len(columns) > 0:
            yield columns


def _construct_filepath(paths, file_suffix):
    if file_suffix:
        path = None
        for p in paths:
            path = p if p.endswith(file_suffix) else path
        return path
    return None


def _setup_datasets(dataset_name, separator, root, split, offset):
<<<<<<< HEAD
=======
    split = check_default_set(split, target_select=('train', 'valid', 'test'))
>>>>>>> upstream/fbsync
    extracted_files = []
    if isinstance(URLS[dataset_name], dict):
        for name, item in URLS[dataset_name].items():
            dataset_tar = download_from_url(item, root=root, hash_value=MD5[dataset_name][name], hash_type='md5')
            extracted_files.extend(extract_archive(dataset_tar))
    elif isinstance(URLS[dataset_name], str):
        dataset_tar = download_from_url(URLS[dataset_name], root=root, hash_value=MD5[dataset_name], hash_type='md5')
        extracted_files.extend(extract_archive(dataset_tar))
    else:
        raise ValueError(
            "URLS for {} has to be in a form of dictionary or string".format(
                dataset_name))

    data_filenames = {
        "train": _construct_filepath(extracted_files, "train.txt"),
        "valid": _construct_filepath(extracted_files, "dev.txt"),
        "test": _construct_filepath(extracted_files, "test.txt")
    }
<<<<<<< HEAD
    return [RawTextIterableDataset(dataset_name, NUM_LINES[dataset_name][item],
                                   _create_data_from_iob(data_filenames[item], separator), offset=offset)
            if data_filenames[item] is not None else None for item in split]


@wrap_split_argument
@add_docstring_header()
def UDPOS(root=".data", split=('train', 'valid', 'test'), offset=0):
    """
=======
    return tuple(RawTextIterableDataset(dataset_name, NUM_LINES[dataset_name][item],
                 _create_data_from_iob(data_filenames[item], separator), offset=offset)
                 if data_filenames[item] is not None else None for item in split)


def UDPOS(root=".data", split=('train', 'valid', 'test'), offset=0):
    """ Universal Dependencies English Web Treebank

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: a string or tuple for the returned datasets (Default: ('train', 'valid', 'test'))
            By default, all the datasets (train, valid, test) are generated.
            Users could also choose any one or two of them,
            for example ('train', 'valid', 'test') or just a string 'train'.
        offset: the number of the starting line. Default: 0

>>>>>>> upstream/fbsync
    Examples:
        >>> from torchtext.experimental.datasets.raw import UDPOS
        >>> train_dataset, valid_dataset, test_dataset = UDPOS()
    """
    return _setup_datasets("UDPOS", "\t", root, split, offset)


<<<<<<< HEAD
@wrap_split_argument
@add_docstring_header()
def CoNLL2000Chunking(root=".data", split=('train', 'test'), offset=0):
    """
=======
def CoNLL2000Chunking(root=".data", split=('train', 'test'), offset=0):
    """ CoNLL 2000 Chunking Dataset

    Separately returns the training and test dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: a string or tuple for the returned datasets (Default: ('train', 'test'))
            By default, both datasets (train, test) are generated. Users could also choose any one or two of them,
            for example ('train', 'test') or just a string 'train'.
        offset: the number of the starting line. Default: 0

>>>>>>> upstream/fbsync
    Examples:
        >>> from torchtext.experimental.datasets.raw import CoNLL2000Chunking
        >>> train_dataset, test_dataset = CoNLL2000Chunking()
    """
    return _setup_datasets("CoNLL2000Chunking", " ", root, split, offset)


DATASETS = {
    "UDPOS": UDPOS,
    "CoNLL2000Chunking": CoNLL2000Chunking
}
<<<<<<< HEAD

=======
>>>>>>> upstream/fbsync
NUM_LINES = {
    "UDPOS": {'train': 12543, 'valid': 2002, 'test': 2077},
    "CoNLL2000Chunking": {'train': 8936, 'test': 2012}
}
<<<<<<< HEAD

=======
>>>>>>> upstream/fbsync
MD5 = {
    "UDPOS": 'bdcac7c52d934656bae1699541424545',
    "CoNLL2000Chunking": {'train': '6969c2903a1f19a83569db643e43dcc8', 'test': 'a916e1c2d83eb3004b38fc6fcd628939'}
}
