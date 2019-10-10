import torch
from .common.torchtext_test_case import TorchtextTestCase

from torchtext.datasets import SNLI, MultiNLI
from torchtext.datasets.nli import ParsedTextField, ShiftReduceField
from torchtext.data import LabelField, Iterator

import shutil


class TestNLI(TorchtextTestCase):

    def test_snli(self):
        batch_size = 4

        # create fields
        TEXT = ParsedTextField()
        TREE = ShiftReduceField()
        LABEL = LabelField()

        # create train/val/test splits
        train, val, test = SNLI.splits(TEXT, LABEL, TREE)

        # check all are SNLI datasets
        assert type(train) == type(val) == type(test) == SNLI

        # check all have correct number of fields
        assert len(train.fields) == len(val.fields) == len(test.fields) == 5

        # check fields are the correct type
        assert type(train.fields['premise']) == ParsedTextField
        assert type(train.fields['premise_transitions']) == ShiftReduceField
        assert type(train.fields['hypothesis']) == ParsedTextField
        assert type(train.fields['hypothesis_transitions']) == ShiftReduceField
        assert type(train.fields['label']) == LabelField

        assert type(val.fields['premise']) == ParsedTextField
        assert type(val.fields['premise_transitions']) == ShiftReduceField
        assert type(val.fields['hypothesis']) == ParsedTextField
        assert type(val.fields['hypothesis_transitions']) == ShiftReduceField
        assert type(val.fields['label']) == LabelField

        assert type(test.fields['premise']) == ParsedTextField
        assert type(test.fields['premise_transitions']) == ShiftReduceField
        assert type(test.fields['hypothesis']) == ParsedTextField
        assert type(test.fields['hypothesis_transitions']) == ShiftReduceField
        assert type(test.fields['label']) == LabelField

        # check each is the correct length
        assert len(train) == 549367
        assert len(val) == 9842
        assert len(test) == 9824

        # build vocabulary
        TEXT.build_vocab(train)
        LABEL.build_vocab(train)

        # ensure vocabulary has been created
        assert hasattr(TEXT, 'vocab')
        assert hasattr(TEXT.vocab, 'itos')
        assert hasattr(TEXT.vocab, 'stoi')

        # create iterators
        train_iter, val_iter, test_iter = Iterator.splits((train, val, test),
                                                          batch_size=batch_size)

        # get a batch to test
        batch = next(iter(train_iter))

        # split premise and hypothesis from tuples to tensors
        premise, premise_transitions = batch.premise
        hypothesis, hypothesis_transitions = batch.hypothesis
        label = batch.label

        # check each is actually a tensor
        assert type(premise) == torch.Tensor
        assert type(premise_transitions) == torch.Tensor
        assert type(hypothesis) == torch.Tensor
        assert type(hypothesis_transitions) == torch.Tensor
        assert type(label) == torch.Tensor

        # check have the correct batch dimension
        assert premise.shape[-1] == batch_size
        assert premise_transitions.shape[-1] == batch_size
        assert hypothesis.shape[-1] == batch_size
        assert hypothesis_transitions.shape[-1] == batch_size
        assert label.shape[-1] == batch_size

        # repeat the same tests with iters instead of split
        train_iter, val_iter, test_iter = SNLI.iters(batch_size=batch_size,
                                                     trees=True)

        # split premise and hypothesis from tuples to tensors
        premise, premise_transitions = batch.premise
        hypothesis, hypothesis_transitions = batch.hypothesis
        label = batch.label

        # check each is actually a tensor
        assert type(premise) == torch.Tensor
        assert type(premise_transitions) == torch.Tensor
        assert type(hypothesis) == torch.Tensor
        assert type(hypothesis_transitions) == torch.Tensor
        assert type(label) == torch.Tensor

        # check have the correct batch dimension
        assert premise.shape[-1] == batch_size
        assert premise_transitions.shape[-1] == batch_size
        assert hypothesis.shape[-1] == batch_size
        assert hypothesis_transitions.shape[-1] == batch_size
        assert label.shape[-1] == batch_size

        # remove downloaded snli directory
        shutil.rmtree('.data/snli', ignore_errors=True)

    def test_multinli(self):
        batch_size = 4

        TEXT = ParsedTextField()
        TREE = ShiftReduceField()
        GENRE = LabelField()
        LABEL = LabelField()

        train, val, test = MultiNLI.splits(TEXT, LABEL, TREE, GENRE)

        assert type(train) == type(val) == type(test) == MultiNLI

        assert len(train.fields) == len(val.fields) == len(test.fields) == 6

        assert type(train.fields['premise']) == ParsedTextField
        assert type(train.fields['premise_transitions']) == ShiftReduceField
        assert type(train.fields['hypothesis']) == ParsedTextField
        assert type(train.fields['hypothesis_transitions']) == ShiftReduceField
        assert type(train.fields['label']) == LabelField
        assert type(train.fields['genre']) == LabelField

        assert type(val.fields['premise']) == ParsedTextField
        assert type(val.fields['premise_transitions']) == ShiftReduceField
        assert type(val.fields['hypothesis']) == ParsedTextField
        assert type(val.fields['hypothesis_transitions']) == ShiftReduceField
        assert type(val.fields['label']) == LabelField
        assert type(val.fields['genre']) == LabelField

        assert type(test.fields['premise']) == ParsedTextField
        assert type(test.fields['premise_transitions']) == ShiftReduceField
        assert type(test.fields['hypothesis']) == ParsedTextField
        assert type(test.fields['hypothesis_transitions']) == ShiftReduceField
        assert type(test.fields['label']) == LabelField
        assert type(test.fields['genre']) == LabelField

        assert len(train) == 392702
        assert len(val) == 9815
        assert len(test) == 9832

        TEXT.build_vocab(train)
        LABEL.build_vocab(train)
        GENRE.build_vocab(train)

        # ensure vocabulary has been created
        assert hasattr(TEXT, 'vocab')
        assert hasattr(TEXT.vocab, 'itos')
        assert hasattr(TEXT.vocab, 'stoi')

        train_iter, val_iter, test_iter = Iterator.splits((train, val, test),
                                                          batch_size=batch_size)

        batch = next(iter(train_iter))

        # split premise and hypothesis from tuples to tensors
        premise, premise_transitions = batch.premise
        hypothesis, hypothesis_transitions = batch.hypothesis
        label = batch.label
        genre = batch.genre

        # check each is actually a tensor
        assert type(premise) == torch.Tensor
        assert type(premise_transitions) == torch.Tensor
        assert type(hypothesis) == torch.Tensor
        assert type(hypothesis_transitions) == torch.Tensor
        assert type(label) == torch.Tensor
        assert type(genre) == torch.Tensor

        # check have the correct batch dimension
        assert premise.shape[-1] == batch_size
        assert premise_transitions.shape[-1] == batch_size
        assert hypothesis.shape[-1] == batch_size
        assert hypothesis_transitions.shape[-1] == batch_size
        assert label.shape[-1] == batch_size
        assert genre.shape[-1] == batch_size

        # repeat the same tests with iters instead of split
        train_iter, val_iter, test_iter = MultiNLI.iters(batch_size=batch_size,
                                                         trees=True)

        # split premise and hypothesis from tuples to tensors
        premise, premise_transitions = batch.premise
        hypothesis, hypothesis_transitions = batch.hypothesis
        label = batch.label

        # check each is actually a tensor
        assert type(premise) == torch.Tensor
        assert type(premise_transitions) == torch.Tensor
        assert type(hypothesis) == torch.Tensor
        assert type(hypothesis_transitions) == torch.Tensor
        assert type(label) == torch.Tensor

        # check have the correct batch dimension
        assert premise.shape[-1] == batch_size
        assert premise_transitions.shape[-1] == batch_size
        assert hypothesis.shape[-1] == batch_size
        assert hypothesis_transitions.shape[-1] == batch_size
        assert label.shape[-1] == batch_size

        # remove downloaded snli directory
        shutil.rmtree('.data/multinli', ignore_errors=True)
