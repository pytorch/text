import torch
from torchtext import vocab

from collections import Counter
c = Counter(['hello', 'world'])
v = vocab.Vocab(c, wv_path='/home/james.bradbury/chainer-research/util/'
                'glove.840B.300d')
print(v.itos)
print(v.vectors)
