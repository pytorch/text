from torchtext import vocab

from collections import Counter
c = Counter(['hello', 'world'])
v = vocab.Vocab(c, vectors=['glove.twitter.27B.200d', 'charngram.100d'])
print(v.itos)
print(v.vectors)
