import torch
import torchtext
from model import TextSentiment
import sys
import argparse

from torchtext.datasets.text_classification import text_normalize
from torchtext.data.utils import generate_ngrams


def predict(text, model, dictionary):
    with torch.no_grad():
        text = torch.tensor([dictionary.get(token, dictionary['<unk>'])
                             for token in text_normalize(text)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict text from stdin given model and dictionary')
    parser.add_argument('model')
    parser.add_argument('dictionary')
    args = parser.parse_args()

    model = torch.load(args.model)
    dictionary = torch.load(args.dictionary)
    for line in sys.stdin:
        print(predict(line, model, dictionary))
