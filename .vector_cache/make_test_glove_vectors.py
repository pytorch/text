
import torch
import numpy as np

if __name__ == '__main__':
    vocab = ['The', 'the', 'hello', 'world', 'Hello', ',', '.']
    dim = 200

    stoi = {word: i for i, word in enumerate(vocab)}

    vectors = torch.Tensor(len(vocab), dim)
    with open('glove.twitter.27B.200d.txt', 'r') as fin:
        for line in fin:
            columns = line.strip().split()
            if columns[0] in stoi:
                vectors[stoi[columns[0]], :] = torch.from_numpy(
                    np.array([float(ele) for ele in columns[1:]])
                )
    torch.save((stoi, vectors, dim), 'glove.test_twitter.27B.200d.pt')

