import torch
import sys
import argparse

from torchtext.datasets.text_classification import text_normalize


def predict(text, model, dictionary):
    r'''
    The predict() function here is used to test the model on a sample text.
    The input text is numericalized with the dictionary and then sent to
    the model for inference.

    Arguments:
        text: a sample text string
        model: the trained model
        dictionary: a vocab object for the information of string-to-index

    '''

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
