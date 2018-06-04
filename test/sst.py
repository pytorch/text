from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText


# Approach 1:
# set up fields
TEXT = data.Field()
LABEL = data.Field(sequential=False)

# make splits for data
train, val, test = datasets.SST.splits(
    TEXT, LABEL, fine_grained=True, train_subtrees=True,
    filter_pred=lambda ex: ex.label != 'neutral')

# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

# build the vocabulary
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.build_vocab(train, vectors=Vectors('wiki.simple.vec', url=url))
LABEL.build_vocab(train)

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))
print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

# make iterator for splits
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=3)

# print batch information
batch = next(iter(train_iter))
print(batch.text)
print(batch.label)

# Approach 2:
TEXT.build_vocab(train, vectors=[GloVe(name='840B', dim='300'), CharNGram(), FastText()])
LABEL.build_vocab(train)

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))
print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

train_iter, val_iter, test_iter = datasets.SST.iters(batch_size=4)

# print batch information
batch = next(iter(train_iter))
print(batch.text)
print(batch.label)

# Approach 3:
f = FastText()
TEXT.build_vocab(train, vectors=f)
TEXT.vocab.extend(f)
LABEL.build_vocab(train)

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))
print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

train_iter, val_iter, test_iter = datasets.SST.iters(batch_size=4)

# print batch information
batch = next(iter(train_iter))
print(batch.text)
print(batch.label)
