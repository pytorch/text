from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

# Define the fields associated with the sequences.
WORD = data.Field(init_token="<bos>", eos_token="<eos>")
UD_TAG = data.Field(init_token="<bos>", eos_token="<eos>")

# Download and the load default data.
train, val, test = datasets.UDPOS.splits(
    fields=(('word', WORD), ('udtag', UD_TAG), (None, None)))

print(train.fields)
print(len(train))
print(vars(train[0]))

# We can also define more than two columns.
WORD = data.Field(init_token="<bos>", eos_token="<eos>")
UD_TAG = data.Field(init_token="<bos>", eos_token="<eos>")
PTB_TAG = data.Field(init_token="<bos>", eos_token="<eos>")

# Load the specified data.
train, val, test = datasets.UDPOS.splits(
    fields=(('word', WORD), ('udtag', UD_TAG), ('ptbtag', PTB_TAG)),
    path=".data/sequence-labeling/en-ud-v2",
    train="en-ud-tag.v2.train.txt",
    validation="en-ud-tag.v2.dev.txt",
    test="en-ud-tag.v2.test.txt")

print(train.fields)
print(len(train))
print(vars(train[0]))

WORD.build_vocab(train.word, min_freq=3)
UD_TAG.build_vocab(train.udtag)
PTB_TAG.build_vocab(train.ptbtag)

print(UD_TAG.vocab.freqs)
print(PTB_TAG.vocab.freqs)

train_iter, val_iter = data.BucketIterator.splits(
    (train, val), batch_size=3, device="cuda:0")

batch = next(iter(train_iter))

print("words", batch.word)
print("udtags", batch.udtag)
print("ptbtags", batch.ptbtag)

# Now lets try both word and character embeddings
WORD = data.Field(init_token="<bos>", eos_token="<eos>")
PTB_TAG = data.Field(init_token="<bos>", eos_token="<eos>")

# We'll use NestedField to tokenize each word into list of chars
CHAR_NESTING = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>")
CHAR = data.NestedField(CHAR_NESTING, init_token="<bos>", eos_token="<eos>")

fields = [(('word', 'char'), (WORD, CHAR)), (None, None), ('ptbtag', PTB_TAG)]
train, val, test = datasets.UDPOS.splits(fields=fields)

print(train.fields)
print(len(train))
print(vars(train[0]))

WORD.build_vocab(train.word, val.word, test.word, vectors=[GloVe(name='6B', dim='300')])
CHAR.build_vocab(train.char, val.char, test.char)
PTB_TAG.build_vocab(train.ptbtag)

print(CHAR.vocab.freqs)
train_iter, val_iter = data.BucketIterator.splits(
    (train, val), batch_size=3)

batch = next(iter(train_iter))

print("words", batch.word)
print("chars", batch.char)
print("ptbtags", batch.ptbtag)

# Using the CoNLL 2000 Chunking dataset:
INPUTS = data.Field(init_token="<bos>", eos_token="<eos>")
CHUNK_TAGS = data.Field(init_token="<bos>", eos_token="<eos>")

train, val, test = datasets.CoNLL2000Chunking.splits(
    fields=(('inputs', INPUTS), (None, None), ('tags', CHUNK_TAGS))
)
print(len(train), len(val), len(test))

# Using the ATIS dataset
TEXT = data.Field(lower=True, batch_first=True)
SLOT = data.Field(batch_first=True, unk_token=None)
INTENT = data.Field(batch_first=True, unk_token=None)

# make splits for data
train, val, test = datasets.ATIS.splits(fields=(('text', TEXT), ('slot', SLOT), ('intent', INTENT)))
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
SLOT.build_vocab(train, val, test)
INTENT.build_vocab(train, val, test)

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))
print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

# make iterator for splits
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_sizes=(32, 256, 256))

# print batch information
batch = next(iter(train_iter))
print(batch.text)
print(batch.slot)
print(batch.intent)
