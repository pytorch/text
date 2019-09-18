from torchtext import data
from torchtext import datasets

# Testing SNLI
print("Run test on SNLI...")
TEXT = datasets.nli.ParsedTextField()
LABEL = data.LabelField()
TREE = datasets.nli.ShiftReduceField()

train, val, test = datasets.SNLI.splits(TEXT, LABEL, TREE)

print("Fields:", train.fields)
print("Number of examples:\n", len(train))
print("First Example instance:\n", vars(train[0]))

TEXT.build_vocab(train)
LABEL.build_vocab(train)

train_iter, val_iter, test_iter = data.Iterator.splits((train, val, test), batch_size=3)

batch = next(iter(train_iter))
print("Numericalize premises:\n", batch.premise)
print("Numericalize hypotheses:\n", batch.hypothesis)
print("Entailment labels:\n", batch.label)

print("Test iters function")
train_iter, val_iter, test_iter = datasets.SNLI.iters(batch_size=4, trees=True)

batch = next(iter(train_iter))
print("Numericalize premises:\n", batch.premise)
print("Numericalize hypotheses:\n", batch.hypothesis)
print("Entailment labels:\n", batch.label)


# Testing MultiNLI
print("Run test on MultiNLI...")
TEXT = datasets.nli.ParsedTextField()
LABEL = data.LabelField()
GENRE = data.LabelField()
TREE = datasets.nli.ShiftReduceField()

train, val, test = datasets.MultiNLI.splits(TEXT, LABEL, TREE, GENRE)

print("Fields:", train.fields)
print("Number of examples:\n", len(train))
print("First Example instance:\n", vars(train[0]))

TEXT.build_vocab(train)
LABEL.build_vocab(train)
GENRE.build_vocab(train, val, test)

train_iter, val_iter, test_iter = data.Iterator.splits((train, val, test), batch_size=3)

batch = next(iter(train_iter))
print("Numericalize premises:\n", batch.premise)
print("Numericalize hypotheses:\n", batch.hypothesis)
print("Entailment labels:\n", batch.label)
print("Genre categories:\n", batch.genre)

print("Test iters function")
train_iter, val_iter, test_iter = datasets.MultiNLI.iters(batch_size=4, trees=True)

batch = next(iter(train_iter))
print("Numericalize premises:\n", batch.premise)
print("Numericalize hypotheses:\n", batch.hypothesis)
print("Entailment labels:\n", batch.label)

# Testing XNLI
print("Run test on XNLI...")
TEXT = data.Field()
LABEL = data.LabelField()
GENRE = data.Field()
LANGUAGE = data.Field()

val, test = datasets.XNLI.splits(TEXT, LABEL, GENRE, LANGUAGE)

print("Fields:", val.fields)
print("Number of examples:\n", len(val))
print("First Example instance:\n", vars(val[0]))

TEXT.build_vocab(val)
LABEL.build_vocab(val)
GENRE.build_vocab(val, test)
LANGUAGE.build_vocab(val, test)

val_iter, test_iter = data.Iterator.splits((val, test), batch_size=3)

batch = next(iter(val_iter))
print("Numericalize premises:\n", batch.premise)
print("Numericalize hypotheses:\n", batch.hypothesis)
print("Entailment labels:\n", batch.label)
print("Genre categories:\n", batch.genre)
print("Languages:\n", batch.genre)