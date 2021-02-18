import io
from torchtext.utils import (download_from_url, extract_archive)
from torchtext.datasets.common import RawTextIterableDataset
from torchtext.datasets.common import wrap_split_argument
from torchtext.datasets.common import add_docstring_header

_URL_BASE_ = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task'

URL = [
    '1/raw/test_2016_flickr.cs.gz',
    '1/raw/test_2016_flickr.de.gz',
    '1/raw/test_2016_flickr.en.gz',
    '1/raw/test_2016_flickr.fr.gz',
    '1/raw/test_2017_flickr.de.gz',
    '1/raw/test_2017_flickr.en.gz',
    '1/raw/test_2017_flickr.fr.gz',
    '1/raw/test_2017_mscoco.de.gz',
    '1/raw/test_2017_mscoco.en.gz',
    '1/raw/test_2017_mscoco.fr.gz',
    '1/raw/test_2018_flickr.en.gz',
    '1/raw/train.cs.gz',
    '1/raw/train.de.gz',
    '1/raw/train.en.gz',
    '1/raw/train.fr.gz',
    '1/raw/val.cs.gz',
    '1/raw/val.de.gz',
    '1/raw/val.en.gz',
    '1/raw/val.fr.gz',
    '2/raw/test_2016.1.de.gz',
    '2/raw/test_2016.1.en.gz',
    '2/raw/test_2016.2.de.gz',
    '2/raw/test_2016.2.en.gz',
    '2/raw/test_2016.3.de.gz',
    '2/raw/test_2016.3.en.gz',
    '2/raw/test_2016.4.de.gz',
    '2/raw/test_2016.4.en.gz',
    '2/raw/test_2016.5.de.gz',
    '2/raw/test_2016.5.en.gz',
    '2/raw/train.1.de.gz',
    '2/raw/train.1.en.gz',
    '2/raw/train.2.de.gz',
    '2/raw/train.2.en.gz',
    '2/raw/train.3.de.gz',
    '2/raw/train.3.en.gz',
    '2/raw/train.4.de.gz',
    '2/raw/train.4.en.gz',
    '2/raw/train.5.de.gz',
    '2/raw/train.5.en.gz',
    '2/raw/val.1.de.gz',
    '2/raw/val.1.en.gz',
    '2/raw/val.2.de.gz',
    '2/raw/val.2.en.gz',
    '2/raw/val.3.de.gz',
    '2/raw/val.3.en.gz',
    '2/raw/val.4.de.gz',
    '2/raw/val.4.en.gz',
    '2/raw/val.5.de.gz',
    '2/raw/val.5.en.gz'
]
URL = [_URL_BASE_ + u for u in URL]

MD5 = [
    '3104872229daa1bef3b401d44dd2220b',
    'efd67d314d98489b716b145475101932',
    'ff2c0fcb4893a13bd73414306bc250ae',
    '08dc7cd4a662f31718412de95ca9bfe3',
    '6a8d5c87f6ae19e3d35681aa6fd16571',
    '005396bac545d880abe6f00bbb7dbbb4',
    'cb09af7d2b501f9112f2d6a59fa1360d',
    'e8cd6ec2bc8a11fc846fa48a46e3d0bb',
    'a7b684e0edbef1d4a23660c8e8e743fd',
    '4995d10954a804d3cdfd907b9fd093e8',
    'a152878809942757a55ce087073486b8',
    'd9a5fc268917725a2b0efce3a0cc8607',
    '81ff90b99829c0cd4b1b587d394afd39',
    '0065d13af80720a55ca8153d126e6627',
    '6cb767741dcad3931f966fefbc05203f',
    '83cdc082f646b769095615384cf5c0ca',
    '6e0e229eb049e3fc99a1ef02fb2d5f91',
    '2b69aa9253948ac9f67e94917272dd40',
    '93fc564584b7e5ba410c761ea5a1c682',
    'ac0c72653c140dd96707212a1baa4278',
    'eec05227daba4bb8f3f8f25b1cb335f4',
    '6dfb42cae4e4fd9a3c40e62ff5398a55',
    '9318fa08c0c0b96114eadb10eb2fc633',
    'ece8cec6b87bf00dd12607f3062dae4c',
    '088ec0765fa213a0eb937a62adfd4996',
    '9a7e7b2dcc33135a32cd621c3b37d2d8',
    '5f7c8d0be0ac739856b47d32a9434998',
    '7d5ef0f069ee2d74dc2fdc6b46cd47fa',
    '713ed720636622a54546d5f14f88b00f',
    '62f36422bfab90fb42a560546b704009',
    'cbf5bfc2147706f228d288e1b18bf4af',
    '540da4566bb6dd35fdbc720218b742b7',
    'bdfe4222f4692ccaa1e3389460f0890e',
    '613eb4a3f0c2b13f0871ced946851b0e',
    '0e1ee2b4145795bd180b193424db204b',
    'd848fe0ae8b9447209fb49c5c31cb3d2',
    '1cff688d1aadef7fdb22e9ad27d6fd2c',
    'abc13b4042f4fef1cdff6de3b6c53b71',
    '3e10289959d0059952511c31df3c7550',
    'b26486ede1d4436d5acf6e38c65bb44d',
    'df57faf5f00d434d2559c021ef55f1aa',
    '16165248083beacebfe18866d5f4f0ae',
    '9077a5127480cc799116384de501bd70',
    '7180780822d4b600eb81c1ccf171c230',
    'c1f697c3b6dfb7305349db34e26b45fc',
    '8edb43c90cae66ec762748a968089b99',
    'acb5ea26a577ceccfae6337181c31716',
    '873a377a348713d3ab84db1fb57cdede',
    '680816e0938fea5cf5331444bc09a4cf'
]

NUM_LINES = {
    'train': 29000,
    'valid': 1014,
    'test': 1000,
}


def _read_text_iterator(path):
    with io.open(path, encoding="utf8") as f:
        for row in f:
            yield row


def _construct_filepaths(paths, src_filename, tgt_filename):
    src_path = None
    tgt_path = None
    for p in paths:
        src_path = p if src_filename in p else src_path
        tgt_path = p if tgt_filename in p else tgt_path
    return (src_path, tgt_path)


@wrap_split_argument
@add_docstring_header()
def Multi30k(root='.data', split=('train', 'valid', 'test'), offset=0,
             train_filenames=("train.de", "train.en"),
             valid_filenames=("val.de", "val.en"),
             test_filenames=("test_2016_flickr.de", "test_2016_flickr.en")):
    """    train_filenames: the source and target filenames for training.
                Default: ('train.de', 'train.en')
            valid_filenames: the source and target filenames for valid.
                Default: ('val.de', 'val.en')
            test_filenames: the source and target filenames for test.
                Default: ('test2016.de', 'test2016.en')

    Examples:
        >>> from torchtext.experimental.datasets.raw import Multi30k
        >>> train_dataset, valid_dataset, test_dataset = Multi30k()

    The available dataset include:
        test_2016_flickr.cs
        test_2016_flickr.de
        test_2016_flickr.en
        test_2016_flickr.fr
        test_2017_flickr.de
        test_2017_flickr.en
        test_2017_flickr.fr
        test_2017_mscoco.de
        test_2017_mscoco.en
        test_2017_mscoco.fr
        test_2018_flickr.en
        train.cs
        train.de
        train.en
        train.fr
        val.cs
        val.de
        val.en
        val.fr
        test_2016.1.de
        test_2016.1.en
        test_2016.2.de
        test_2016.2.en
        test_2016.3.de
        test_2016.3.en
        test_2016.4.de
        test_2016.4.en
        test_2016.5.de
        test_2016.5.en
        train.1.de
        train.1.en
        train.2.de
        train.2.en
        train.3.de
        train.3.en
        train.4.de
        train.4.en
        train.5.de
        train.5.en
        val.1.de
        val.1.en
        val.2.de
        val.2.en
        val.3.de
        val.3.en
        val.4.de
        val.4.en
        val.5.de
        val.5.en
    """
    if not isinstance(train_filenames, tuple) and not isinstance(valid_filenames, tuple) \
            and not isinstance(test_filenames, tuple):
        raise ValueError("All filenames must be tuples")
    src_train, tgt_train = train_filenames
    src_eval, tgt_eval = valid_filenames
    src_test, tgt_test = test_filenames

    extracted_files = []  # list of paths to the extracted files

    for idx, f in enumerate(URL):
        dataset_tar = download_from_url(
            f, root=root, hash_value=MD5[idx], hash_type='md5')
        extracted_files.extend(extract_archive(dataset_tar))

    file_archives = extracted_files

    data_filenames = {
        "train": _construct_filepaths(file_archives, src_train, tgt_train),
        "valid": _construct_filepaths(file_archives, src_eval, tgt_eval),
        "test": _construct_filepaths(file_archives, src_test, tgt_test)
    }

    for key in data_filenames.keys():
        if len(data_filenames[key]) == 0 or data_filenames[key] is None:
            raise FileNotFoundError(
                "Files are not found for data type {}".format(key))

    datasets = []
    for key in split:
        src_data_iter = _read_text_iterator(data_filenames[key][0])
        tgt_data_iter = _read_text_iterator(data_filenames[key][1])

        def _iter(src_data_iter, tgt_data_iter):
            for item in zip(src_data_iter, tgt_data_iter):
                yield item

        datasets.append(
            RawTextIterableDataset("Multi30k", NUM_LINES[key], _iter(src_data_iter, tgt_data_iter), offset=offset))

    return datasets
