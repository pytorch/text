# [WIP] torch-text

This repository consists of:

- [torchtext.data](#data) : Generic data loaders, abstractions, and iterators for text
- [torchtext.datasets](#datasets) : Pre-built loaders for common NLP datasets
- (maybe) torchtext.models : Model definitions and pre-trained models for popular NLP examples
(though the situation is not the same as vision, where people can download a pretrained ImageNet model and immediately
make it useful for other tasks -- it might make more sense to leave NLP models in the torch/examples repo)

# Data

The data module provides the following:

- Ability to describe declaratively how to load a custom NLP dataset that's in a "normal" format:
```python
pos = data.TabularDataset(
    path='data/pos/pos_wsj_train.tsv', format='tsv',
    fields=[('text', data.Field()),
            ('labels', data.Field())])

sentiment = data.TabularDataset(
    path='data/sentiment/train.json', format='json',
    fields=[{'sentence_tokenized': ('text', data.Field(time_series=True)),
             'sentiment_gold': ('labels', data.Field(time_series=False))}])
```
- Ability to define a preprocessing pipeline:
```python
src = data.Field(tokenize=my_custom_tokenizer)
trg = data.Field(tokenize=my_custom_tokenizer)
mt_train = datasets.TranslationDataset(
    path='data/mt/wmt16-ende.train', exts=('.en', '.de'),
    fields=(src, trg))
```
- Batching, padding, and numericalizing (including building a vocabulary object):
```python
# continuing from above
mt_dev = data.TranslationDataset(
    path='data/mt/newstest2014', exts=('.en', '.de'),
    fields=(src, trg))
src.build_vocab(mt_train, max_size=80000)
trg.build_vocab(mt_train, max_size=40000)
# mt_dev shares the fields, so it shares their vocab objects

train_iter = data.BucketIterator(
    batch_size=32, mt_train,
    sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))
# usage
>>>next(train_iter)
<data.Batch(batch_size=32, src=[LongTensor (32, 25)], trg=[LongTensor (32, 28)])>
```
- Wrapper for dataset splits (train, validation, test):
```python
TEXT = data.Field()
LABELS = data.Field()

train, val, test = data.TabularDataset.splits(
    path='/data/pos_wsj/pos_wsj', train='_train.tsv',
    validation='_dev.tsv', test='_test.tsv', format='tsv',
    fields=[('text', TEXT), ('labels', LABELS)])

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_sizes=(16, 256, 256),
    sort_key=lambda x: len(x.text), device=0)

TEXT.build_vocab(train)
LABELS.build_vocab(train)
```

# Datasets

Some datasets it would be useful to have built in:

- bAbI and successors from FAIR
- SST (done) and IMDb sentiment
- SNLI (done)
- Penn Treebank (for language modeling (done) and parsing)
- WMT and/or IWSLT machine translation
- SQuAD

See the "test" directory for examples of dataset usage.
