import os
from torchtext.data.datasets_utils import (
    _download_extract_validate,
    _RawTextIterableDataset,
    _wrap_split_argument,
    _create_dataset_directory,
    _read_text_iterator,
)

URL = {
    'train': r'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz',
    'valid': r'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz',
    'test': r'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz',
}

MD5 = {
    'train': '20140d013d05dd9a72dfde46478663ba05737ce983f478f960c1123c6671be5e',
    'valid': 'a7aa20e9ebd5ba5adce7909498b94410996040857154dab029851af3a866da8c',
    'test': '0681be16a532912288a91ddd573594fbdd57c0fbb81486eff7c55247e35326c2',
}

_EXTRACTED_FILES_INFO = {
    'train': {
        'file_prefix': 'train',
        'md5': {
            'de': '695df46f6fd14567e69970408a2c129a50e778a910ecb1585a92eb25b2c7accc',
            'en': '4b4d37e774976ef44fecca1738cdeb3b3ba384851a59a755b9c5e6aa7d87b13c',
        },
    },
    'valid': {
        'file_prefix': 'val',
        'md5': {
            'de': 'fd0fc009db2446cfc12d96a382aff0d3122cb47577b352d0f7e0bb3a38e2e552',
            'en': '40cd20974079d9afb0e3d27c659a8e059cc2fcf850b4bc23ede13fc36dd8a865',
        },
    },
    'test': {
        'file_prefix': 'test',
        'md5': {
            'de': 'c1d2f544471a7387e37d15f1adf075ff7d6fe57a30840bb969281ae102d24cb1',
            'en': '399a4382932c1aadd3ceb9bef1008d388a64c76d4ae4e9d4728c6f4301cac182',
        },
    },
}

NUM_LINES = {
    'train': 29000,
    'valid': 1014,
    'test': 1000,
}

DATASET_NAME = "Multi30k"


@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'valid', 'test'))
def Multi30k(root, split, language_pair=('de', 'en')):
    """Multi30k dataset

    Reference: http://www.statmt.org/wmt16/multimodal-task.html#task1

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (‘train’, ‘valid’, ‘test’)
        language_pair: tuple or list containing src and tgt language. Available options are ('de','en') and ('en', 'de')
    """

    assert (len(language_pair) == 2), 'language_pair must contain only 2 elements: src and tgt language respectively'
    assert (tuple(sorted(language_pair)) == ('de', 'en')), "language_pair must be either ('de','en') or ('en', 'de')"

    downloaded_file = os.path.basename(URL[split])

    src_path = _download_extract_validate(root, URL[split], MD5[split],
                                          os.path.join(root, downloaded_file),
                                          os.path.join(root, _EXTRACTED_FILES_INFO[split]['file_prefix'] + '.' + language_pair[0]),
                                          _EXTRACTED_FILES_INFO[split]['md5'][language_pair[0]])
    trg_path = _download_extract_validate(root, URL[split], MD5[split],
                                          os.path.join(root, downloaded_file),
                                          os.path.join(root, _EXTRACTED_FILES_INFO[split]['file_prefix'] + '.' + language_pair[1]),
                                          _EXTRACTED_FILES_INFO[split]['md5'][language_pair[1]])

    src_data_iter = _read_text_iterator(src_path)
    trg_data_iter = _read_text_iterator(trg_path)

    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split], zip(src_data_iter, trg_data_iter))
