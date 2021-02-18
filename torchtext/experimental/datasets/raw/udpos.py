from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header

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


def _construct_filepath(paths, file_suffix):
    if file_suffix:
        path = None
        for p in paths:
            path = p if p.endswith(file_suffix) else path
        return path
    return None


@wrap_split_argument
@add_docstring_header()
def UDPOS(root='.data', split=('train', 'valid', 'test'), offset=0):
    datasets = []
    for item in split:
        dataset_tar = download_from_url(item, root=root, hash_value=MD5[item], hash_type='md5')
        extracted_files = extract_archive(dataset_tar)
        if split == "valid":
            data_filenames = _construct_filepath(extracted_files, "dev.txt")
        else:
            data_filenames = _construct_filepath(extracted_files, split + ".txt")
        datasets.append(RawTextIterableDataset("UDPOS", NUM_LINES[item],
                                               _create_data_from_iob(data_filenames[item], " "), offset=offset))
    return datasets
