from argparse import ArgumentParser

import torch
from torchtext import transforms


def main(args):
    tokenizer_file = args.tokenizer_file
    sentence = "The green grasshopper jumped over the fence"

    # create tokenizer object
    encoder_json = "gpt2_bpe_encoder.json"
    bpe_vocab = "gpt2_bpe_vocab.bpe"
    tokenizer = transforms.GPT2BPETokenizer(encoder_json_path=encoder_json, vocab_bpe_path=bpe_vocab)

    # script and save tokenizer
    tokenizer = torch.jit.script(tokenizer)
    print(tokenizer(sentence))
    torch.jit.save(tokenizer, tokenizer_file)

    # load saved tokenizer and verify outputs match
    t = torch.jit.load(tokenizer_file)
    print(t(sentence))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tokenizer-file", default="tokenizer.pt", type=str)
    main(parser.parse_args())
