import argparse
import sys

import torch
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from torchtext.experimental.transforms import load_sp_model, PRETRAINED_SP_MODEL, SentencePieceTokenizer
from torchtext.utils import download_from_url


def predict(text, model, dictionary, tokenizer, ngrams):
    r"""
    The predict() function here is used to test the model on a sample text.
    The input text is numericalized with the vocab and then sent to
    the model for inference.

    Args:
        text: a sample text string
        model: the trained model
        dictionary: a vocab object for the information of string-to-index
        tokenizer: tokenizer object to split text into tokens
        ngrams: the number of ngrams.
    """
    with torch.no_grad():
        text = torch.tensor(dictionary(list(ngrams_iterator(tokenizer(text), ngrams))))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict text from stdin given model and dictionary")
    parser.add_argument("model", help="the path for model")
    parser.add_argument("dictionary", help="the path for dictionary")
    parser.add_argument("--ngrams", type=int, default=2, help="ngrams (default=2)")
    parser.add_argument(
        "--use-sp-tokenizer", type=bool, default=False, help="use sentencepiece tokenizer (default=False)"
    )
    args = parser.parse_args()

    model = torch.load(args.model)
    dictionary = torch.load(args.dictionary)
    if args.use_sp_tokenizer:
        sp_model_path = download_from_url(PRETRAINED_SP_MODEL["text_unigram_15000"])
        sp_model = load_sp_model(sp_model_path)
        tokenizer = SentencePieceTokenizer(sp_model)
    else:
        tokenizer = get_tokenizer("basic_english")
    for line in sys.stdin:
        print(predict(line, model, dictionary, tokenizer, args.ngrams))
