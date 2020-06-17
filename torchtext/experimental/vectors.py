import csv
import os

import torch
from torch import Tensor
import torch.nn as nn

from torchtext.utils import (
    download_from_url,
    extract_archive
)

def fast_text(language="en", unk_tensor=None):
    r"""Create a fast text Vectors object.

    Args:
        language (str): the language to use for FastText.
        unk_tensor (int): a 1d tensors representing the vector associated with an unknown token

    Returns:
        Vectors: a Vectors object.

    """
    url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec'.format(language)
    downloaded_file_path = download_from_url(url)
    csv_file_name, _ext = os.path.splitext(downloaded_file_path)
    csv_file_path = csv_file_name + '.csv'

    # skip if csv file already exists 
    if not os.path.exists(csv_file_path):
        with open(downloaded_file_path, 'r') as f1:
            with open(csv_file_path, "w") as f2:
                cnt = 0
                for line in f1:
                    print(line)
                    # f2.write(line)
                    if cnt == 4:
                        break
                    cnt += 1

            f2.close()
        f1.close()

    # file_object = open(csv_file_path, 'r')
    # return vectors_from_file_object(csv_file_path)

    # print(csv_file_name, _ext)
    # extracted_paths = extract_archive(downloaded_file_path)
    # print(downloaded_file_path)



    # for path in extracted_paths:
    #     print(path)


def vectors_from_file_object(file_like_object, unk_tensor=None):
    r"""Create a Vectors object from a csv file like object.

    Note that the tensor corresponding to each vector is of type `torch.float`.

    Format for csv file:
        token1,num1 num2 num3
        token2,num4 num5 num6
        ...
        token_n,num_m num_j num_k

    Args:
        file_like_object (FileObject): a file like object to read data from.
        unk_tensor (int): a 1d tensors representing the vector associated with an unknown token

    Returns:
        Vectors: a Vectors object.

    """
    readCSV = csv.reader(file_like_object, delimiter=',')

    tokens = []
    vectors = []
    for row in readCSV:
        tokens.append(row[0])
        vectors.append(torch.tensor([float(c) for c in row[1].split()], dtype=torch.float))

    return Vectors(tokens, vectors, unk_tensor=unk_tensor)


class Vectors(nn.Module):
    r"""Creates a vectors object which maps tokens to vectors.

    Arguments:
        tokens (List[str]): a list of tokens.
        vectors (List[torch.Tensor]): a list of 1d tensors representing the vector associated with each token.
        unk_tensor (torch.Tensor): a 1d tensors representing the vector associated with an unknown token.

    Raises:
        ValueError: if `vectors` is empty and a default `unk_tensor` isn't provided.
        RuntimeError: if `tokens` and `vectors` have different sizes or `tokens` has duplicates.
        TypeError: if all tensors within`vectors` are not of data type `torch.float`.
    """

    def __init__(self, tokens, vectors, unk_tensor=None):
        super(Vectors, self).__init__()

        if unk_tensor is None and not vectors:
            raise ValueError("The vectors list is empty and a default unk_tensor wasn't provided.")

        if not all(vector.dtype == torch.float for vector in vectors):
            raise TypeError("All tensors within `vectors` should be of data type `torch.float`.")

        unk_tensor = unk_tensor if unk_tensor is not None else torch.zeros(vectors[0].size(), dtype=torch.float)

        self.vectors = torch.classes.torchtext.Vectors(tokens, vectors, unk_tensor)

    @torch.jit.export
    def __getitem__(self, token: str) -> Tensor:
        r"""
        Args:
            token (str): the token used to lookup the corresponding vector.
        Returns:
            vector (Tensor): a tensor (the vector) corresponding to the associated token.
        """
        return self.vectors.GetItem(token)

    @torch.jit.export
    def __setitem__(self, token: str, vector: Tensor):
        r"""
        Args:
            token (str): the token used to lookup the corresponding vector.
            vector (Tensor): a 1d tensor representing a vector associated with the token.

        Raises:
            TypeError: if `vector` is not of data type `torch.float`.
        """
        if vector.dtype != torch.float:
            raise TypeError("`vector` should be of data type `torch.float` but it's of type " + vector.dtype)

        self.vectors.AddItem(token, vector.float())
