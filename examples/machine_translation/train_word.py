import argparse
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from embedding import WordCharCNNEmbedding
from model import Attention, Decoder, Encoder, Seq2Seq
from torchtext.data.metrics import bleu_score
from torchtext.experimental.datasets import Multi30k
from torchtext.vocab import Vocab
from utils import (count_parameters, epoch_time, pad_chars, pad_words,
                   seed_everything)


def train(
    model: nn.Module,
    iterator: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    clip: float,
    trg_vocab: Vocab,
    device: torch.device,
):
    model.train()

    epoch_loss = 0
    bleu = 0.0
    for _, batch in tqdm(enumerate(iterator), total=len(iterator)):
        # Need to convert to [seq_len x batch x ...] size
        src = batch[0].transpose(0, 1).to(device)
        trg = batch[1].transpose(0, 1).to(device)
        optimizer.zero_grad()

        output = model(src, trg)
        # Convert prediction to words
        true = trg.transpose(0, 1).unsqueeze(1)
        true = [
            [[trg_vocab.itos[word_idx] for word_idx in comb if trg_vocab.itos[word_idx] != " "] for comb in sent]
            for sent in true
        ]
        pred = output.transpose(0, 1).argmax(-1)
        pred = [[trg_vocab.itos[word_idx] for word_idx in sent if trg_vocab.itos[word_idx] != " "] for sent in pred]

        output = output.reshape(-1, output.shape[-1])
        trg = trg.reshape(-1)
        loss = criterion(output, trg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        bleu += bleu_score(pred, true)

    return (epoch_loss / len(iterator)), (bleu / len(iterator))


def evaluate(model: nn.Module, iterator: DataLoader, criterion: nn.Module, trg_vocab: Vocab, device: torch.device):
    model.eval()

    epoch_loss = 0
    bleu = 0.0
    with torch.no_grad():
        for _, batch in tqdm(enumerate(iterator), total=len(iterator)):
            # Need to convert to [seq_len x batch x ...] size
            src = batch[0].transpose(0, 1).to(device)
            trg = batch[1].transpose(0, 1).to(device)

            output = model(src, trg)
            # Convert prediction to words
            true = trg.transpose(0, 1).unsqueeze(1)
            true = [
                [[trg_vocab.itos[word_idx] for word_idx in comb if trg_vocab.itos[word_idx] != " "] for comb in sent]
                for sent in true
            ]
            pred = output.transpose(0, 1).argmax(-1)
            pred = [[trg_vocab.itos[word_idx] for word_idx in sent if trg_vocab.itos[word_idx] != " "] for sent in pred]

            output = output.reshape(-1, output.shape[-1])
            trg = trg.reshape(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
            bleu += bleu_score(pred, true)
    return (epoch_loss / len(iterator)), (bleu / len(iterator))


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    padded_src_batch = pad_words(src_batch)
    padded_tgt_batch = pad_words(tgt_batch)
    return (padded_src_batch, padded_tgt_batch)


def main(args):
    # Ensure reproducibility
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size
    # train_dataset, val_dataset, test_dataset = get_dataset(args.dataset)
    train_dataset, val_dataset, test_dataset = Multi30k()
    train_iterator = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    valid_iterator = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_iterator = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    input_dim = len(train_dataset.vocab[0])
    output_dim = len(train_dataset.vocab[1])

    enc_emb = nn.Embedding(input_dim, args.enc_emb_dim, padding_idx=train_dataset.vocab[0].stoi["<pad>"])
    enc = Encoder(input_dim, args.enc_emb_dim, args.enc_hid_dim, args.dec_hid_dim, args.enc_dropout, enc_emb)
    attn = Attention(args.enc_hid_dim, args.dec_hid_dim, args.attn_dim)
    dec_emb = nn.Embedding(output_dim, args.dec_emb_dim, padding_idx=train_dataset.vocab[1].stoi["<pad>"])
    dec = Decoder(output_dim, args.dec_emb_dim, args.enc_hid_dim, args.dec_hid_dim, args.dec_dropout, attn, dec_emb)
    model = Seq2Seq(enc, dec, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab[1].stoi["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"The model has {count_parameters(model):,} trainable parameters")
    n_epochs = args.epochs
    clip = args.clip

    best_valid_loss = float("inf")

    for epoch in range(n_epochs):

        start_time = time.time()

        train_loss, train_bleu_score = train(
            model, train_iterator, optimizer, criterion, clip, train_dataset.vocab[1], device
        )
        valid_loss, valid_bleu_score = evaluate(model, valid_iterator, criterion, train_dataset.vocab[1], device)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} |  Train BLEU: {train_bleu_score:7.3f}"
        )
        print(
            f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} |  Val. BLEU: {valid_bleu_score:7.3f}"
        )

    test_loss, test_bleu_score = evaluate(model, test_iterator, criterion, train_dataset.vocab[2], device)

    print(f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |  Test BLEU: {test_bleu_score:7.3f}")

    if args.save:
        print("Saving model to {}".format(args.save))
        torch.save(model.to("cpu"), args.save)

    if args.save_vocab:
        print("Save vocab to {}".format(args.save_vocab))
        torch.save(train_dataset.vocab, args.save_vocab)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Experimental Seq2seq for Machine Translation")
    parser.add_argument("--enc_emb_dim", type=int, default=300, help="size of encoder char-composed embeddings")
    parser.add_argument("--dec_emb_dim", type=int, default=300, help="size of decoder char-composed embeddings")
    parser.add_argument("--enc_hid_dim", type=int, default=64, help="size of encoder hidden units")
    parser.add_argument("--dec_hid_dim", type=int, default=64, help="size of decoder hidden units")
    parser.add_argument("--attn_dim", type=int, default=8, help="size of attention weights")
    parser.add_argument("--enc_dropout", type=float, default=0.5, help="dropout applied to encoder")
    parser.add_argument("--dec_dropout", type=float, default=0.5, help="dropout applied to decoder")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--clip", type=float, default=1, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=10, help="upper epoch limit")
    parser.add_argument("--batch_size", type=int, default=128, metavar="N", help="batch size")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--save", type=str, default="mt_seq2seq.pt", help="path to save the final model")
    parser.add_argument("--save_vocab", type=str, default="torchtext_mt_vocab.pt", help="path to save the vocab")
    parser.add_argument("--dataset", type=str, default="Multi30k", help="dataset used for MLM task")
    args = parser.parse_args()

    main(args)
