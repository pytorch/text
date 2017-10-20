[![Build Status](https://travis-ci.org/pytorch/text.svg?branch=master)](https://travis-ci.org/pytorch/text)
[![codecov](https://codecov.io/gh/pytorch/text/branch/master/graph/badge.svg)](https://codecov.io/gh/pytorch/text)

# torchtext

This repository consists of:

- [torchtext.data](#data) : Generic data loaders, abstractions, and iterators for text (including vocabulary and word vectors)
- [torchtext.datasets](#datasets) : Pre-built loaders for common NLP datasets

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
    fields={'sentence_tokenized': ('text', data.Field(sequential=True)),
             'sentiment_gold': ('labels', data.Field(sequential=False))})
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
    dataset=mt_train, batch_size=32, 
    sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))
# usage
>>>next(iter(train_iter))
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

The datasets module currently contains:

- Sentiment analysis: SST and IMDb
- Question classification: TREC
- Entailment: SNLI
- Language modeling: WikiText-2
- Machine Translation: Multi30k, IWSLT, WMT14

Others are planned or a work in progress:

- Question answering: SQuAD

See the "test" directory for examples of dataset usage.
