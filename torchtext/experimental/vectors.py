import torch


class Vectors(object):
    """Creates a vectors object.

    Arguments:
        tokens (List[str]):: a list of tokens.
        vectors (List[torch.Tensor]): a list of 1d tensors representing the vector associated with each token

    """

    def __init__(self, tokens, vectors):
        self.unk_tensor = torch.zeros(vectors[0].size())
        self.vectors = torch.classes.torchtext.Vectors(tokens, vectors)

    def __getitem__(self, token):
        if self.vectors.TokenExists(token):
            return self.vectors.GetItem(token)
        else:
            return self.unk_tensor
