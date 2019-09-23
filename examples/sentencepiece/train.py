import torch
import os
import logging
import argparse
from torch.utils.data import DataLoader
import example_bpm_dataset as text_classification
import time
from torch.utils.data.dataset import random_split
from torchtext.datasets.text_classification import LABELS
from model import TextSentiment

r"""
This example is similar to the one in examples/text_classification.
The only difference is the dataset, which applies SentencePiece as encoder.
The subword method in SentencePiece was tested based on YelpReviewFull and
we are able to reproduce the results from fastText.
"""


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=batch_size, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)


def test_func(test_model, data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=batch_size, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = test_model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Apply SentencePiece to text classification dataset.')
    parser.add_argument('--dataset', default='YelpReviewFull',
                        help='dataset name')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='num epochs (default=10)')
    parser.add_argument('--embed-dim', type=int, default=64,
                        help='embed dim. (default=32)')
    parser.add_argument('--vocab-size', type=int, default=20000,
                        help='vocab size in sentencepiece model (default=20000)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size (default=16)')
    parser.add_argument('--split-ratio', type=float, default=0.95,
                        help='train/valid split ratio (default=0.95)')
    parser.add_argument('--lr', type=float, default=4.0,
                        help='learning rate (default=4.0)')
    parser.add_argument('--lr-gamma', type=float, default=0.8,
                        help='lr decayed by gamma every step (default=0.8)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='num of workers (default=1)')
    parser.add_argument('--data-directory', default='.data',
                        help='data directory (default=.data)')
    parser.add_argument('--logging-level', default='WARNING',
                        help='logging level (default=WARNING)')
    args = parser.parse_args()

    r"""
    Load the dataset
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=getattr(logging, args.logging_level))

    if not os.path.exists(args.data_directory):
        print("Creating directory {}".format(args.data_directory))
        os.mkdir(args.data_directory)

    train_data, test_data = text_classification.setup_datasets(args.dataset,
                                                               root='.data',
                                                               vocab_size=args.vocab_size)
    NUM_CLASS = len(LABELS[args.dataset])
    batch_size = args.batch_size

    r"""
    Load the model
    """
    model = TextSentiment(args.vocab_size, args.embed_dim, NUM_CLASS).to(device)
    best_model = None  # Save the best model for test purpose
    best_val_loss = float("inf")

    r"""
    Set up the training loop and train the model
    """
    min_valid_loss = float('inf')

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.lr_gamma)

    train_len = int(len(train_data) * args.split_ratio)
    sub_train_, sub_valid_ = random_split(train_data,
                                          [train_len, len(train_data) - train_len])

    for epoch in range(args.num_epochs):

        start_time = time.time()
        train_loss, train_acc = train_func(sub_train_)
        valid_loss, valid_acc = test_func(model, sub_valid_)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' % (epoch + 1),
              ' | time in %d minutes, %d seconds' % (mins, secs))
        print('\tLoss:\t %.4f(train)\t|\tAcc:\t %.1f'
              % (train_loss, train_acc * 100), '%')
        print('\tLoss:\t %.4f(valid)\t|\tAcc:\t %.1f'
              % (valid_loss, valid_acc * 100), '%')

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_model = model

    print('Checking the results of test dataset...')
    # Use the best model so far for testing
    test_loss, test_acc = test_func(best_model, test_data)
    print('\tLoss:\t %.4f(valid)\t|\tAcc:\t %.1f'
          % (test_loss, test_acc * 100), '%')
