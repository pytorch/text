import logging
import torch
import io
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.utils import generate_sp_tokenizer
from os import path
from torchtext.datasets.text_classification import URLS
import sentencepiece as spm
from torchtext.data.transforms import SentencePieceTransform


def _create_data_with_sp_transform(sp_transform, data_path):

    data = []
    labels = []
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            corpus = ' '.join(row[1:])
            token_ids = sp_transform(corpus)
            label = int(row[0]) - 1
            data.append((label, torch.tensor(token_ids)))
            labels.append(label)
    return data, set(labels)


class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        """Initiate text-classification dataset.
           The original one is here (torchtext/datasets/text_classification.py).
        """

        super(TextClassificationDataset, self).__init__()
        self._data = data
        self._labels = labels

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_labels(self):
        return self._labels


def setup_datasets(dataset_name, root='.data', vocab_size=20000, include_unk=False):
    dataset_tar = download_from_url(URLS[dataset_name], root=root)
    extracted_files = extract_archive(dataset_tar)

    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname

    # generate sentencepiece  pretrained tokenizer
    if not path.exists('m_user.model'):
        logging.info('Generate SentencePiece pretrained tokenizer...')
        generate_sp_tokenizer(train_csv_path, vocab_size)

    sp_transform = SentencePieceTransform("m_user.model")
    train_data, train_labels = _create_data_with_sp_transform(sp_transform,
                                                              train_csv_path)
    test_data, test_labels = _create_data_with_sp_transform(sp_transform,
                                                            test_csv_path)

    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (TextClassificationDataset(train_data, train_labels),
            TextClassificationDataset(test_data, test_labels))
