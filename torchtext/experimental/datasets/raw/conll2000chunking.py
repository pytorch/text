from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header

URL = {
    'train': "https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz",
    'test': "https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz",
}

MD5 = {
    'train': "6969c2903a1f19a83569db643e43dcc8",
    'test': "a916e1c2d83eb3004b38fc6fcd628939",
}

NUM_LINES = {
    'train': 8936,
    'test': 2012,
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


@wrap_split_argument
@add_docstring_header()
def CoNLL2000Chunking(root='.data', split=('train', 'test'), offset=0):
    extracted_files = []
    for name, item in URL.items():
        dataset_tar = download_from_url(item, root=root, hash_value=MD5[name], hash_type='md5')
        extracted_files.extend(extract_archive(dataset_tar))

    data_filenames = {
        "train": _construct_filepath(extracted_files, "train.txt"),
        "valid": _construct_filepath(extracted_files, "dev.txt"),
        "test": _construct_filepath(extracted_files, "test.txt")
    }
    return [RawTextIterableDataset("CoNLL2000Chunking", NUM_LINES[item],
                                   _create_data_from_iob(data_filenames[item], " "), offset=offset)
            if data_filenames[item] is not None else None for item in split]
