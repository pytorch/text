import torch
import torchtext
from model import TextSentiment
import sys

from torchtext.datasets.text_classification import URLS
from torchtext.data.utils import generate_ngrams

def predict(text, model, dictionary):
    with torch.no_grad():
        text = torchtext.datasets.text_classification.text_normalize(text)
        text = torch.tensor([dictionary.get(token, dictionary['UNK']) for token in text])
        output = model(text, torch.tensor([0]))
        print(output)
        # TODO: Why is this 0?
        return output.argmax(1)

if __name__ == "__main__":
    dictionary = torch.load("/tmp/asdf/dictionary.torch")
    model = torch.load("/tmp/asdf/model.torch")
    print("Done loading")
    for line in sys.stdin:
        print(predict(line, model, dictionary))
