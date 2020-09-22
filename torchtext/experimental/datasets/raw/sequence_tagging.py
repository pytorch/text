import torch

from torchtext.utils import download_from_url, extract_archive

URLS = {
    "UDPOS":
    'https://bitbucket.org/sivareddyg/public/downloads/en-ud-v2.zip',
    "CoNLL2000Chunking": [
        'https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz',
        'https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz'
    ]
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


def _setup_datasets(dataset_name, separator, root=".data"):

    extracted_files = []
    if isinstance(URLS[dataset_name], list):
        for f in URLS[dataset_name]:
            dataset_tar = download_from_url(f, root=root)
            extracted_files.extend(extract_archive(dataset_tar))
    elif isinstance(URLS[dataset_name], str):
        dataset_tar = download_from_url(URLS[dataset_name], root=root)
        extracted_files.extend(extract_archive(dataset_tar))
    else:
        raise ValueError(
            "URLS for {} has to be in a form or list or string".format(
                dataset_name))

    data_filenames = {
        "train": _construct_filepath(extracted_files, "train.txt"),
        "valid": _construct_filepath(extracted_files, "dev.txt"),
        "test": _construct_filepath(extracted_files, "test.txt")
    }

    datasets = []
    for key in data_filenames.keys():
        if data_filenames[key] is not None:
            datasets.append(
                RawTextIterableDataset(dataset_name, NUM_LINES[dataset_name],
                    _create_data_from_iob(data_filenames[key], separator)))
        else:
            datasets.append(None)

    return datasets


def UDPOS(*args, **kwargs):
    """ Universal Dependencies English Web Treebank

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> from torchtext.datasets.raw import UDPOS
        >>> train_dataset, valid_dataset, test_dataset = UDPOS()
    """
    return _setup_datasets(*(("UDPOS", "\t") + args), **kwargs)


def CoNLL2000Chunking(*args, **kwargs):
    """ CoNLL 2000 Chunking Dataset

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"

    Examples:
        >>> from torchtext.datasets.raw import CoNLL2000Chunking
        >>> train_dataset, valid_dataset, test_dataset = CoNLL2000Chunking()
    """
    return _setup_datasets(*(("CoNLL2000Chunking", " ") + args), **kwargs)


DATASETS = {
    "UDPOS": UDPOS,
    "CoNLL2000Chunking": CoNLL2000Chunking
}

NUM_LINES = {
    "UDPOS": 12543,
    "CoNLL2000Chunking": 8936
}
