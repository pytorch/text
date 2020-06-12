import csv
import torch


def vectors_from_csv_file(file_like_object, unk_tensor=None):
    r"""Create a Vectors object from a csv file like object.

    Args:
        file_like_object (FileObject): a file like object to read data from.
        unk_tensor (int): a 1d tensors representing the vector associated with an unknown token
    Returns:
        Vectors: a Vectors object.
    """
    readCSV = csv.reader(file_like_object, delimiter=',')

    tuples = list(map(lambda x: (x[0], torch.tensor([float(c) for c in x[1].split()], dtype=torch.float)), readCSV))
    words = [pair[0] for pair in tuples]
    vectors = [pair[1] for pair in tuples]

    return Vectors(words, vectors, unk_tensor=unk_tensor)


class Vectors(object):
    r"""Creates a vectors object which maps tokens to vectors.

    Arguments:
        tokens (List[str]):: a list of tokens.
        vectors (List[torch.Tensor]): a list of 1d tensors representing the vector associated with each token.
        unk_tensor (torch.Tensor): a 1d tensors representing the vector associated with an unknown token.

    """

    def __init__(self, tokens, vectors, unk_tensor=None):
        unk_tensor = unk_tensor if unk_tensor is not None else torch.zeros(vectors[0].size())
        self.vectors = torch.classes.torchtext.Vectors(tokens, vectors, unk_tensor)

    def __getitem__(self, token):
        return self.vectors.GetItem(token)
