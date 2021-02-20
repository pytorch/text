from torchtext.utils import download_from_url, extract_archive
from torchtext.data.datasets_utils import RawTextIterableDataset
from torchtext.data.datasets_utils import wrap_split_argument
from torchtext.data.datasets_utils import add_docstring_header
from torchtext.data.datasets_utils import find_match

URL = 'https://bitbucket.org/sivareddyg/public/downloads/en-ud-v2.zip'

MD5 = 'bdcac7c52d934656bae1699541424545'

NUM_LINES = {
    'train': 12543,
    'valid': 2002,
    'test': 2077,
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


@add_docstring_header()
@wrap_split_argument(('train', 'valid', 'test'))
def UDPOS(root, split):
    dataset_tar = download_from_url(URL, root=root, hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)
    if split == 'valid':
        path = find_match("dev.txt", extracted_files)
    else:
        path = find_match(split + ".txt", extracted_files)
    return RawTextIterableDataset("UDPOS", NUM_LINES[split],
                                  _create_data_from_iob(path))
