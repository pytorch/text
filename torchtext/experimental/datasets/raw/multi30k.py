import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.experimental.datasets.raw.common import RawTextIterableDataset
from torchtext.experimental.datasets.raw.common import wrap_split_argument
from torchtext.experimental.datasets.raw.common import add_docstring_header
import os

URL = [
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.cs.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.fr.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2017_flickr.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2017_flickr.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2017_flickr.fr.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2017_mscoco.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2017_mscoco.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2017_mscoco.fr.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2018_flickr.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.cs.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.fr.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.cs.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.fr.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.1.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.1.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.2.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.2.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.3.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.3.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.4.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.4.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.5.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/test_2016.5.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.1.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.1.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.2.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.2.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.3.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.3.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.4.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.4.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.5.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/train.5.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.1.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.1.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.2.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.2.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.3.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.3.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.4.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.4.en.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.5.de.gz',
    'https://raw.githubusercontent.com/multi30k/dataset/master/data/task2/raw/val.5.en.gz'
]

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


@wrap_split_argument
@add_docstring_header()
def Multi30k(root='.data', split=('train', 'valid', 'test'), offset=0):
    pass

