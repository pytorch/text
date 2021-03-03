import io
import os
from torchtext.utils import (download_from_url, extract_archive)
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _wrap_split_argument

SUPPORTED_DATASETS = {
    'task1': {
        'cs': {
            'test_2016_flickr': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/test_2016_flickr.cs.gz',
                'MD5': '3104872229daa1bef3b401d44dd2220b',
                'NUM_LINES': 1000
            },
            'train': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/train.cs.gz',
                'MD5': 'd9a5fc268917725a2b0efce3a0cc8607',
                'NUM_LINES': 29000
            },
            'val': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/val.cs.gz',
                'MD5': '83cdc082f646b769095615384cf5c0ca',
                'NUM_LINES': 1014
            }
        },
        'de': {
            'test_2016_flickr': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/test_2016_flickr.de.gz',
                'MD5': 'efd67d314d98489b716b145475101932',
                'NUM_LINES': 1000
            },
            'test_2017_flickr': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/test_2017_flickr.de.gz',
                'MD5': '6a8d5c87f6ae19e3d35681aa6fd16571',
                'NUM_LINES': 1000
            },
            'test_2017_mscoco': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/test_2017_mscoco.de.gz',
                'MD5': 'e8cd6ec2bc8a11fc846fa48a46e3d0bb',
                'NUM_LINES': 461
            },
            'train': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/train.de.gz',
                'MD5': '81ff90b99829c0cd4b1b587d394afd39',
                'NUM_LINES': 29000
            },
            'val': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/val.de.gz',
                'MD5': '6e0e229eb049e3fc99a1ef02fb2d5f91',
                'NUM_LINES': 1014
            }
        },
        'en': {
            'test_2016_flickr': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/test_2016_flickr.en.gz',
                'MD5': 'ff2c0fcb4893a13bd73414306bc250ae',
                'NUM_LINES': 1000
            },
            'test_2017_flickr': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/test_2017_flickr.en.gz',
                'MD5': '005396bac545d880abe6f00bbb7dbbb4',
                'NUM_LINES': 1000
            },
            'test_2017_mscoco': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/test_2017_mscoco.en.gz',
                'MD5': 'a7b684e0edbef1d4a23660c8e8e743fd',
                'NUM_LINES': 461
            },
            'test_2018_flickr': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/test_2018_flickr.en.gz',
                'MD5': 'a152878809942757a55ce087073486b8',
                'NUM_LINES': 1071
            },
            'train': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/train.en.gz',
                'MD5': '0065d13af80720a55ca8153d126e6627',
                'NUM_LINES': 29000
            },
            'val': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/val.en.gz',
                'MD5': '2b69aa9253948ac9f67e94917272dd40',
                'NUM_LINES': 1014
            }
        },
        'fr': {
            'test_2016_flickr': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/test_2016_flickr.fr.gz',
                'MD5': '08dc7cd4a662f31718412de95ca9bfe3',
                'NUM_LINES': 1000
            },
            'test_2017_flickr': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/test_2017_flickr.fr.gz',
                'MD5': 'cb09af7d2b501f9112f2d6a59fa1360d',
                'NUM_LINES': 1000
            },
            'test_2017_mscoco': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/test_2017_mscoco.fr.gz',
                'MD5': '4995d10954a804d3cdfd907b9fd093e8',
                'NUM_LINES': 461
            },
            'train': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/train.fr.gz',
                'MD5': '6cb767741dcad3931f966fefbc05203f',
                'NUM_LINES': 29000
            },
            'val': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task1/raw/val.fr.gz',
                'MD5': '93fc564584b7e5ba410c761ea5a1c682',
                'NUM_LINES': 1014
            }
        }
    },
    'task2': {
        'de': {
            'test_2016.1': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/test_2016.1.de.gz',
                'MD5': 'ac0c72653c140dd96707212a1baa4278',
                'NUM_LINES': 1000
            },
            'test_2016.2': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/test_2016.2.de.gz',
                'MD5': '6dfb42cae4e4fd9a3c40e62ff5398a55',
                'NUM_LINES': 1000
            },
            'test_2016.3': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/test_2016.3.de.gz',
                'MD5': 'ece8cec6b87bf00dd12607f3062dae4c',
                'NUM_LINES': 1000
            },
            'test_2016.4': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/test_2016.4.de.gz',
                'MD5': '9a7e7b2dcc33135a32cd621c3b37d2d8',
                'NUM_LINES': 1000
            },
            'test_2016.5': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/test_2016.5.de.gz',
                'MD5': '7d5ef0f069ee2d74dc2fdc6b46cd47fa',
                'NUM_LINES': 1000
            },
            'train.1': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/train.1.de.gz',
                'MD5': '62f36422bfab90fb42a560546b704009',
                'NUM_LINES': 29000
            },
            'train.2': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/train.2.de.gz',
                'MD5': '540da4566bb6dd35fdbc720218b742b7',
                'NUM_LINES': 29000
            },
            'train.3': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/train.3.de.gz',
                'MD5': '613eb4a3f0c2b13f0871ced946851b0e',
                'NUM_LINES': 29000
            },
            'train.4': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/train.4.de.gz',
                'MD5': 'd848fe0ae8b9447209fb49c5c31cb3d2',
                'NUM_LINES': 29000
            },
            'train.5': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/train.5.de.gz',
                'MD5': 'abc13b4042f4fef1cdff6de3b6c53b71',
                'NUM_LINES': 29000
            },
            'val.1': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/val.1.de.gz',
                'MD5': 'b26486ede1d4436d5acf6e38c65bb44d',
                'NUM_LINES': 1014
            },
            'val.2': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/val.2.de.gz',
                'MD5': '16165248083beacebfe18866d5f4f0ae',
                'NUM_LINES': 1014
            },
            'val.3': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/val.3.de.gz',
                'MD5': '7180780822d4b600eb81c1ccf171c230',
                'NUM_LINES': 1014
            },
            'val.4': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/val.4.de.gz',
                'MD5': '8edb43c90cae66ec762748a968089b99',
                'NUM_LINES': 1014
            },
            'val.5': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/val.5.de.gz',
                'MD5': '873a377a348713d3ab84db1fb57cdede',
                'NUM_LINES': 1014
            }
        },
        'en': {
            'test_2016.1': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/test_2016.1.en.gz',
                'MD5': 'eec05227daba4bb8f3f8f25b1cb335f4',
                'NUM_LINES': 1000
            },
            'test_2016.2': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/test_2016.2.en.gz',
                'MD5': '9318fa08c0c0b96114eadb10eb2fc633',
                'NUM_LINES': 1000
            },
            'test_2016.3': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/test_2016.3.en.gz',
                'MD5': '088ec0765fa213a0eb937a62adfd4996',
                'NUM_LINES': 1000
            },
            'test_2016.4': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/test_2016.4.en.gz',
                'MD5': '5f7c8d0be0ac739856b47d32a9434998',
                'NUM_LINES': 1000
            },
            'test_2016.5': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/test_2016.5.en.gz',
                'MD5': '713ed720636622a54546d5f14f88b00f',
                'NUM_LINES': 1000
            },
            'train.1': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/train.1.en.gz',
                'MD5': 'cbf5bfc2147706f228d288e1b18bf4af',
                'NUM_LINES': 29000
            },
            'train.2': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/train.2.en.gz',
                'MD5': 'bdfe4222f4692ccaa1e3389460f0890e',
                'NUM_LINES': 29000
            },
            'train.3': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/train.3.en.gz',
                'MD5': '0e1ee2b4145795bd180b193424db204b',
                'NUM_LINES': 29000
            },
            'train.4': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/train.4.en.gz',
                'MD5': '1cff688d1aadef7fdb22e9ad27d6fd2c',
                'NUM_LINES': 29000
            },
            'train.5': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/train.5.en.gz',
                'MD5': '3e10289959d0059952511c31df3c7550',
                'NUM_LINES': 29000
            },
            'val.1': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/val.1.en.gz',
                'MD5': 'df57faf5f00d434d2559c021ef55f1aa',
                'NUM_LINES': 1014
            },
            'val.2': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/val.2.en.gz',
                'MD5': '9077a5127480cc799116384de501bd70',
                'NUM_LINES': 1014
            },
            'val.3': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/val.3.en.gz',
                'MD5': 'c1f697c3b6dfb7305349db34e26b45fc',
                'NUM_LINES': 1014
            },
            'val.4': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/val.4.en.gz',
                'MD5': 'acb5ea26a577ceccfae6337181c31716',
                'NUM_LINES': 1014
            },
            'val.5': {
                'URL':
                    'https://raw.githubusercontent.com/multi30k/dataset/master/'
                    'data/task2/raw/val.5.en.gz',
                'MD5': '680816e0938fea5cf5331444bc09a4cf',
                'NUM_LINES': 1014
            }
        }
    }
}


URL = {'train': [], 'valid': [], 'test': []}
MD5 = {'train': [], 'valid': [], 'test': []}
NUM_LINES = {'train': [], 'valid': [], 'test': []}

for task in SUPPORTED_DATASETS:
    for language in SUPPORTED_DATASETS[task]:
        for data in SUPPORTED_DATASETS[task][language]:
            if 'train' in data:
                k = 'train'
            elif 'val' in data:
                k = 'valid'
            else:
                k = 'test'
            URL[k].append(SUPPORTED_DATASETS[task][language][data]['URL'])
            MD5[k].append(SUPPORTED_DATASETS[task][language][data]['MD5'])
            NUM_LINES[k].append(SUPPORTED_DATASETS[task][language][data]['NUM_LINES'])


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


@_wrap_split_argument(('train', 'valid', 'test'))
def Multi30k(root, split,
             task='task1',
             language_pair=('de', 'en'),
             train_set="train",
             valid_set="val",
             test_set="test_2016_flickr"):
    """Multi30k Dataset

    The available datasets include following:

    **Language pairs (task1)**:

    +-----+-----+-----+-----+-----+
    |     |'en' |'cs' |'de' |'fr' |
    +-----+-----+-----+-----+-----+
    |'en' |     |   x |  x  |  x  |
    +-----+-----+-----+-----+-----+
    |'cs' |  x  |     |  x  |  x  |
    +-----+-----+-----+-----+-----+
    |'de' |  x  |   x |     |  x  |
    +-----+-----+-----+-----+-----+
    |'fr' |  x  |   x |  x  |     |
    +-----+-----+-----+-----+-----+

    **Language pairs (task2)**:

    +-----+-----+-----+
    |     |'en' |'de' |
    +-----+-----+-----+
    |'en' |     |   x |
    +-----+-----+-----+
    |'de' |  x  |     |
    +-----+-----+-----+

    For additional details refer to source: https://github.com/multi30k/dataset

    Args:
        root: Directory where the datasets are saved. Default: ".data"
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (‘train’, ‘valid’, ‘test’)
        task: Indicate the task
        language_pair: tuple or list containing src and tgt language
        train_set: A string to identify train set.
        valid_set: A string to identify validation set.
        test_set: A string to identify test set.

    Examples:
        >>> from torchtext.experimental.datasets.raw import Multi30k
        >>> train_iter, valid_iter, test_iter = Multi30k()
        >>> src_sentence, tgt_sentence = next(train_iter)
    """

    if task not in SUPPORTED_DATASETS.keys():
        raise ValueError('task {} is not supported. Valid options are {}'.
                         format(task, SUPPORTED_DATASETS.keys()))

    assert (len(language_pair) == 2), 'language_pair must contain only 2 elements: src and tgt language respectively'

    if language_pair[0] not in SUPPORTED_DATASETS[task].keys():
        raise ValueError("Source language '{}' is not supported. Valid options for task '{}' are {}".
                         format(language_pair[0], task, list(SUPPORTED_DATASETS[task].keys())))

    if language_pair[1] not in SUPPORTED_DATASETS[task].keys():
        raise ValueError("Target language '{}' is not supported. Valid options for task '{}' are {}".
                         format(language_pair[1], task, list(SUPPORTED_DATASETS[task].keys())))

    if train_set not in SUPPORTED_DATASETS[task][language_pair[0]].keys() or 'train' not in train_set:
        raise ValueError("'{}' is not a valid train set identifier. valid options for task '{}' and language pair {} are {}".
                         format(train_set, task, language_pair, [k for k in SUPPORTED_DATASETS[task][language_pair[0]].keys() if 'train' in k]))

    if valid_set not in SUPPORTED_DATASETS[task][language_pair[0]].keys() or 'val' not in valid_set:
        raise ValueError("'{}' is not a valid valid set identifier. valid options for task '{}' and language pair {} are {}".
                         format(valid_set, task, language_pair, [k for k in SUPPORTED_DATASETS[task][language_pair[0]].keys() if 'val' in k]))

    if test_set not in SUPPORTED_DATASETS[task][language_pair[0]].keys() or 'test' not in test_set:
        raise ValueError("'{}' is not a valid test set identifier. valid options for task '{}' and language pair {} are {}".
                         format(test_set, task, language_pair, [k for k in SUPPORTED_DATASETS[task][language_pair[0]].keys() if 'test' in k]))

    train_filenames = ["{}.{}".format(
        train_set, language_pair[0]), "{}.{}".format(train_set,
                                                     language_pair[1])]
    valid_filenames = ["{}.{}".format(
        valid_set, language_pair[0]), "{}.{}".format(valid_set,
                                                     language_pair[1])]
    test_filenames = ["{}.{}".format(
        test_set, language_pair[0]), "{}.{}".format(test_set,
                                                    language_pair[1])]

    if split == 'train':
        src_file, tgt_file = train_filenames
    elif split == 'valid':
        src_file, tgt_file = valid_filenames
    else:
        src_file, tgt_file = test_filenames

    extracted_files = []  # list of paths to the extracted files

    current_url = []
    current_md5 = []

    current_filenames = [src_file, tgt_file]
    for url, md5 in zip(URL[split], MD5[split]):
        if any(f in url for f in current_filenames):
            current_url.append(url)
            current_md5.append(md5)

    for url, md5 in zip(current_url, current_md5):
        dataset_tar = download_from_url(
            url, path=os.path.join(root, os.path.basename(url)), root=root, hash_value=md5, hash_type='md5')
        extracted_files.extend(extract_archive(dataset_tar))

    file_archives = extracted_files

    data_filenames = {
        split: _construct_filepaths(file_archives, src_file, tgt_file),
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

    set_identifier = {
        'train': train_set,
        'valid': valid_set,
        'test': test_set,
    }

    return _RawTextIterableDataset("Multi30k",
                                   SUPPORTED_DATASETS[task][language_pair[0]][set_identifier[split]]['NUM_LINES'],
                                   _iter(src_data_iter, tgt_data_iter))
