import os
import io
from torchtext.utils import (download_from_url, extract_archive)
from torchtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
)

URL = 'https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8'

_PATH = 'wmt16_en_de.tar.gz'

MD5 = '874ab6bbfe9c21ec987ed1b9347f95ec'

NUM_LINES = {
    'newstest2010.tok.bpe.32000': 2489,
    'newstest2012': 3003,
    'newstest2010.tok': 2489,
    'newstest2016': 2999,
    'newstest2014.tok': 3003,
    'newstest2009': 2525,
    'newstest2015.tok.bpe.32000': 2169,
    'newstest2016.tok': 2999,
    'newstest2011.tok.bpe.32000': 3003,
    'newstest2012.tok': 3003,
    'newstest2013': 3000,
    'newstest2014.tok.bpe.32000': 3003,
    'newstest2011.tok': 3003,
    'newstest2011': 3003,
    'newstest2015.tok': 2169,
    'newstest2012.tok.bpe.32000': 3003,
    'newstest2015': 2169,
    'newstest2016.tok.bpe.32000': 2999,
    'newstest2009.tok.bpe.32000': 2525,
    'newstest2014': 3003,
    'newstest2009.tok': 2525,
    'newstest2013.tok.bpe.32000': 3000,
    'newstest2013.tok': 3000,
    'newstest2010': 2489,
    'train.tok.clean.bpe.32000': 4500966
}


def _read_text_iterator(path):
    with io.open(path, encoding="utf8") as f:
        for row in f:
            yield row


def _construct_filenames(filename, languages):
    filenames = []
    for lang in languages:
        filenames.append(filename + "." + lang)
    return filenames


def _construct_filepaths(paths, src_filename, tgt_filename):
    src_path = None
    tgt_path = None
    for p in paths:
        src_path = p if src_filename in p else src_path
        tgt_path = p if tgt_filename in p else tgt_path
    return (src_path, tgt_path)


@_wrap_split_argument(('train', 'valid', 'test'))
def WMT14(root, split,
          language_pair=('de', 'en'),
          train_set='train.tok.clean.bpe.32000',
          valid_set='newstest2013.tok.bpe.32000',
          test_set='newstest2014.tok.bpe.32000'):
    """WMT14 Dataset

    The available datasets include following:

    **Language pairs**:

    +-----+-----+-----+
    |     |'en' |'de' |
    +-----+-----+-----+
    |'en' |     |   x |
    +-----+-----+-----+
    |'de' |  x  |     |
    +-----+-----+-----+


    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (‘train’, ‘valid’, ‘test’)
        language_pair: tuple or list containing src and tgt language
        train_set: A string to identify train set.
        valid_set: A string to identify validation set.
        test_set: A string to identify test set.

    Examples:
        >>> from torchtext.datasets import WMT14
        >>> train_iter, valid_iter, test_iter = WMT14()
        >>> src_sentence, tgt_sentence = next(train_iter)
    """

    supported_language = ['en', 'de']
    supported_train_set = [s for s in NUM_LINES if 'train' in s]
    supported_valid_set = [s for s in NUM_LINES if 'test' in s]
    supported_test_set = [s for s in NUM_LINES if 'test' in s]

    assert (len(language_pair) == 2), 'language_pair must contain only 2 elements: src and tgt language respectively'

    if language_pair[0] not in supported_language:
        raise ValueError("Source language '{}' is not supported. Valid options are {}".
                         format(language_pair[0], supported_language))

    if language_pair[1] not in supported_language:
        raise ValueError("Target language '{}' is not supported. Valid options are {}".
                         format(language_pair[1], supported_language))

    if train_set not in supported_train_set:
        raise ValueError("'{}' is not a valid train set identifier. valid options are {}".
                         format(train_set, supported_train_set))

    if valid_set not in supported_valid_set:
        raise ValueError("'{}' is not a valid valid set identifier. valid options are {}".
                         format(valid_set, supported_valid_set))

    if test_set not in supported_test_set:
        raise ValueError("'{}' is not a valid valid set identifier. valid options are {}".
                         format(test_set, supported_test_set))

    train_filenames = '{}.{}'.format(train_set, language_pair[0]), '{}.{}'.format(train_set, language_pair[1])
    valid_filenames = '{}.{}'.format(valid_set, language_pair[0]), '{}.{}'.format(valid_set, language_pair[1])
    test_filenames = '{}.{}'.format(test_set, language_pair[0]), '{}.{}'.format(test_set, language_pair[1])

    if split == 'train':
        src_file, tgt_file = train_filenames
    elif split == 'valid':
        src_file, tgt_file = valid_filenames
    else:
        src_file, tgt_file = test_filenames

    root = os.path.join(root, 'wmt14')
    dataset_tar = download_from_url(URL, root=root, hash_value=MD5, path=os.path.join(root, _PATH), hash_type='md5')
    extracted_files = extract_archive(dataset_tar)

    data_filenames = {
        split: _construct_filepaths(extracted_files, src_file, tgt_file),
    }

    for key in data_filenames:
        if len(data_filenames[key]) == 0 or data_filenames[key] is None:
            raise FileNotFoundError(
                "Files are not found for data type {}".format(key))

    assert data_filenames[split][0] is not None, "Internal Error: File not found for reading"
    assert data_filenames[split][1] is not None, "Internal Error: File not found for reading"
    src_data_iter = _read_text_iterator(data_filenames[split][0])
    tgt_data_iter = _read_text_iterator(data_filenames[split][1])

    def _iter(src_data_iter, tgt_data_iter):
        for item in zip(src_data_iter, tgt_data_iter):
            yield item

    return _RawTextIterableDataset("WMT14", NUM_LINES[os.path.splitext(src_file)[0]], _iter(src_data_iter, tgt_data_iter))
