import torch
import io
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.datasets.text_classification import URLS
from torchtext.datasets import text_classification
from torchtext.experimental.transforms import (
    sentencepiece_processor,
    pretrained_sp_model,
)


def _create_data_with_sp_transform(data_path):

    data = []
    labels = []
    spm_path = pretrained_sp_model['text_unigram_15000']
    text_pipeline = sentencepiece_processor(download_from_url(spm_path))
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            corpus = ' '.join(row[1:])
            token_ids = text_pipeline(corpus)
            label = int(row[0]) - 1
            data.append((label, torch.tensor(token_ids)))
            labels.append(label)
    return data, set(labels)


def setup_datasets(dataset_name, root='.data', vocab_size=20000, include_unk=False):
    dataset_tar = download_from_url(URLS[dataset_name], root=root)
    extracted_files = extract_archive(dataset_tar)

    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname
    train_data, train_labels = _create_data_with_sp_transform(train_csv_path)
    test_data, test_labels = _create_data_with_sp_transform(test_csv_path)

    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (text_classification.TextClassificationDataset(None, train_data, train_labels),
            text_classification.TextClassificationDataset(None, test_data, test_labels))
