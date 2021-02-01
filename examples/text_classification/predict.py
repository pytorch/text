import torch
import sys
import argparse
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator


def predict(text, model, dictionary, ngrams):
    r"""
    The predict() function here is used to test the model on a sample text.
    The input text is numericalized with the vocab and then sent to
    the model for inference.

    Args:
        text: a sample text string
        model: the trained model
        dictionary: a vocab object for the information of string-to-index
        ngrams: the number of ngrams.
    """
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([dictionary[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict text from stdin given model and dictionary')
    parser.add_argument('model', help='the path for model')
    parser.add_argument('dictionary', help='the path for dictionary')
    parser.add_argument('--ngrams', type=int, default=2,
                        help='ngrams (default=2)')
    args = parser.parse_args()

    model = torch.load(args.model)
    dictionary = torch.load(args.dictionary)
    for line in sys.stdin:
        print(predict(line, model, dictionary, args.ngrams))
