import os
import logging
import argparse

import torch
import sys

from torchtext.datasets import text_classification
from torch.utils.data import DataLoader

from model import TextSentiment

r"""
This file shows the training process of the text classification model.
"""


def generate_batch(batch):
    r"""
    Since the text entries have different lengths, a custom function
    generate_batch() is used to generate data batches and offsets,
    which are compatible with EmbeddingBag. The function is passed
    to 'collate_fn' in torch.utils.data.DataLoader. The input to
    'collate_fn' is a list of tensors with the size of batch_size,
    and the 'collate_fn' function packs them into a mini-batch.
    Pay attention here and make sure that 'collate_fn' is declared
    as a top level def. This ensures that the function is available
    in each worker.

    Output:
        text: the text entries in the data_batch are packed into a list and
            concatenated as a single tensor for the input of nn.EmbeddingBag.
        offsets: the offsets is a tensor of delimiters to represent the beginning
            index of the individual sequence in the text tensor.
        cls: a tensor saving the labels of individual text entries.
    """
    def generate_offsets(data_batch):
        offsets = [0]
        for entry in data_batch:
            offsets.append(offsets[-1] + len(entry))
        offsets = torch.tensor(offsets[:-1])
        return offsets

    cls = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = generate_offsets(text)
    text = torch.cat(text)
    return text, offsets, cls


r"""
torch.utils.data.DataLoader is recommended for PyTorch users to load data.
We use DataLoader here to load datasets and send it to the train()
and text() functions.

"""


def train(lr_, num_epoch, data_):
    r"""
    We use a SGD optimizer to train the model here and the learning rate
    decreases linearly with the progress of the training process.

    Arguments:
        lr_: learning rate
        num_epoch: the number of epoches for training the model
        data_: the data used to train the model
    """

    data = DataLoader(data_, batch_size=batch_size, shuffle=True,
                      collate_fn=generate_batch, num_workers=args.num_workers)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.lr_gamma)
    num_lines = num_epochs * len(data)
    for epoch in range(num_epochs):
        for i, (text, offsets, cls) in enumerate(data):
            optimizer.zero_grad()
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss.backward()
            optimizer.step()
            processed_lines = i + len(data) * epoch
            progress = processed_lines / float(num_lines)
            if processed_lines % 1024:
                sys.stderr.write(
                    "\rProgress: {:3.0f}% lr: {:3.3f} loss: {:3.3f}".format(
                        progress * 100, scheduler.get_lr()[0], loss))
        # Adjust the learning rate
        scheduler.step()
    print("")


def test(data_):
    r"""
    Arguments:
        data_: the data used to train the model
    """
    data = DataLoader(data_, batch_size=batch_size, collate_fn=generate_batch)
    total_accuracy = []
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            accuracy = (output.argmax(1) == cls).float().mean().item()
            total_accuracy.append(accuracy)
    print("Test - Accuracy: {}".format(sum(total_accuracy) / len(total_accuracy)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a text classification model on text classification datasets.')
    parser.add_argument('dataset', choices=text_classification.DATASETS)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=4.0)
    parser.add_argument('--lr-gamma', type=float, default=0.8)
    parser.add_argument('--ngrams', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--data', default='.data')
    parser.add_argument('--save-model-path')
    parser.add_argument('--logging-level', default='WARNING')
    args = parser.parse_args()

    num_epochs = args.num_epochs
    embed_dim = args.embed_dim
    batch_size = args.batch_size
    lr = args.lr
    device = args.device
    data = args.data

    logging.basicConfig(level=getattr(logging, args.logging_level))

    if not os.path.exists(data):
        print("Creating directory {}".format(data))
        os.mkdir(data)

    train_dataset, test_dataset = text_classification.DATASETS[args.dataset](
        root=data, ngrams=args.ngrams)

    model = TextSentiment(len(train_dataset.get_vocab()),
                          embed_dim, len(train_dataset.get_labels())).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    train(lr, num_epochs, train_dataset)
    test(test_dataset)

    if args.save_model_path:
        print("Saving model to {}".format(args.save_model_path))
        torch.save(model.to('cpu'), args.save_model_path)
