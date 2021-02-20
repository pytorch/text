from torchtext.utils import download_from_url, extract_archive
from torchtext.data.datasets_utils import RawTextIterableDataset
from torchtext.data.datasets_utils import wrap_split_argument
from torchtext.data.datasets_utils import add_docstring_header
from torchtext.data.datasets_utils import find_match

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


def _create_data_from_iob(data_path, separator):
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


@add_docstring_header()
@wrap_split_argument(('train', 'test'))
def CoNLL2000Chunking(root, split):
    dataset_tar = download_from_url(URL[split], root=root, hash_value=MD5[split], hash_type='md5')
    extracted_files = extract_archive(dataset_tar)
    data_filename = find_match(split + ".txt", extracted_files)
    return RawTextIterableDataset("CoNLL2000Chunking", NUM_LINES[split],
                                  _create_data_from_iob(data_filename, " "))
