#!/user/bin/env python3
# Note that all the tests in this module require dataset (either network access or cached)
import os
import glob
import shutil
import torchtext.data as data
from torchtext.datasets import AG_NEWS
import torch
from ..common.torchtext_test_case import TorchtextTestCase


def conditional_remove(f):
    for path in glob.glob(f):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


class TestDataset(TorchtextTestCase):
    def test_wikitext2_legacy(self):
        from torchtext.datasets import WikiText2
        # smoke test to ensure wikitext2 works properly

        # NOTE
        # test_wikitext2 and test_wikitext2_legacy have some cache incompatibility.
        # Keeping one's cache make the other fail. So we need to clean up the cache dir
        cachedir = os.path.join(self.project_root, ".data", "wikitext-2")
        conditional_remove(cachedir)

        ds = WikiText2
        TEXT = data.Field(lower=True, batch_first=True)
        train, valid, test = ds.splits(TEXT)
        TEXT.build_vocab(train)
        train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
            (train, valid, test), batch_size=3, bptt_len=30)

        train_iter, valid_iter, test_iter = ds.iters(batch_size=4,
                                                     bptt_len=30)

        conditional_remove(cachedir)

    def test_wikitext2(self):
        from torchtext.experimental.datasets import WikiText2
        # smoke test to ensure wikitext2 works properly

        # NOTE
        # test_wikitext2 and test_wikitext2_legacy have some cache incompatibility.
        # Keeping one's cache make the other fail. So we need to clean up the cache dir
        cachedir = os.path.join(self.project_root, ".data", "wikitext-2")
        conditional_remove(cachedir)
        cachefile = os.path.join(self.project_root, ".data", "wikitext-2-v1.zip")
        conditional_remove(cachefile)

        train_dataset, valid_dataset, test_dataset = WikiText2()
        self.assertEqual(len(train_dataset), 2049990)
        self.assertEqual(len(test_dataset), 241859)
        self.assertEqual(len(valid_dataset), 214417)

        vocab = train_dataset.get_vocab()
        tokens_ids = [vocab[token] for token in 'the player characters rest'.split()]
        self.assertEqual(tokens_ids, [2, 286, 503, 700])

        conditional_remove(cachedir)
        conditional_remove(cachefile)

    def test_penntreebank_legacy(self):
        from torchtext.datasets import PennTreebank
        # smoke test to ensure penn treebank works properly
        TEXT = data.Field(lower=True, batch_first=True)
        ds = PennTreebank
        train, valid, test = ds.splits(TEXT)
        TEXT.build_vocab(train)
        train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
            (train, valid, test), batch_size=3, bptt_len=30)

        train_iter, valid_iter, test_iter = ds.iters(batch_size=4,
                                                     bptt_len=30)

    def test_penntreebank(self):
        from torchtext.experimental.datasets import PennTreebank
        # smoke test to ensure penn treebank works properly
        train_dataset, valid_dataset, test_dataset = PennTreebank()
        self.assertEqual(len(train_dataset), 924412)
        self.assertEqual(len(test_dataset), 82114)
        self.assertEqual(len(valid_dataset), 73339)

        vocab = train_dataset.get_vocab()
        tokens_ids = [vocab[token] for token in 'the player characters rest'.split()]
        self.assertEqual(tokens_ids, [2, 2550, 3344, 1125])

    def test_text_classification(self):
        # smoke test to ensure ag_news dataset works properly

        datadir = os.path.join(self.project_root, ".data")
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        ag_news_train, ag_news_test = AG_NEWS(root=datadir, ngrams=3)
        self.assertEqual(len(ag_news_train), 120000)
        self.assertEqual(len(ag_news_test), 7600)
        self.assertEqual(ag_news_train[-1][1][:10],
                         torch.tensor([3525, 319, 4053, 34, 5407, 3607, 70, 6798, 10599, 4053]).long())
        self.assertEqual(ag_news_test[-1][1][:10],
                         torch.tensor([2351, 758, 96, 38581, 2351, 220, 5, 396, 3, 14786]).long())

    def test_imdb(self):
        from torchtext.experimental.datasets import IMDB
        from torchtext.vocab import Vocab
        # smoke test to ensure imdb works properly
        train_dataset, test_dataset = IMDB()
        self.assertEqual(len(train_dataset), 25000)
        self.assertEqual(len(test_dataset), 25000)
        self.assertEqual(train_dataset[0][1][:10],
                         torch.tensor([13, 1568, 13, 246, 35468, 43, 64, 398, 1135, 92]).long())
        self.assertEqual(train_dataset[-1][1][:10],
                         torch.tensor([2, 71, 4555, 194, 3328, 15144, 42, 227, 148, 8]).long())
        self.assertEqual(test_dataset[0][1][:10],
                         torch.tensor([13, 125, 1051, 5, 246, 1652, 8, 277, 66, 20]).long())
        self.assertEqual(test_dataset[-1][1][:10],
                         torch.tensor([13, 1035, 14, 21, 28, 2, 1051, 1275, 1008, 3]).long())

        # Test API with a vocab input object
        old_vocab = train_dataset.get_vocab()
        new_vocab = Vocab(counter=old_vocab.freqs, max_size=2500)
        new_train_data, new_test_data = IMDB(vocab=new_vocab)

    def test_multi30k(self):
        from torchtext.experimental.datasets import Multi30k
        # smoke test to ensure multi30k works properly
        train_dataset, valid_dataset, test_dataset = Multi30k()
        self.assertEqual(len(train_dataset), 29000)
        self.assertEqual(len(valid_dataset), 1014)
        self.assertEqual(len(test_dataset), 1000)

        de_vocab, en_vocab = train_dataset.get_vocab()
        de_tokens_ids = [
            de_vocab[token] for token in
            'Zwei Männer verpacken Donuts in Kunststofffolie'.split()
        ]
        print(de_tokens_ids)
        self.assertEqual(de_tokens_ids, [19, 29, 18703, 4448, 5, 6240])

        en_tokens_ids = [
            en_vocab[token] for token in
            'Two young White males are outside near many bushes'.split()
        ]
        self.assertEqual(en_tokens_ids,
                         [17, 23, 1167, 806, 15, 55, 82, 334, 1337])

        datafile = os.path.join(self.project_root, ".data", "train*")
        conditional_remove(datafile)
        datafile = os.path.join(self.project_root, ".data", "val*")
        conditional_remove(datafile)
        datafile = os.path.join(self.project_root, ".data", "test*")
        conditional_remove(datafile)
        datafile = os.path.join(self.project_root, ".data",
                                "multi30k_task*.tar.gz")
        conditional_remove(datafile)

    def test_udpos_sequence_tagging(self):
        from torchtext.experimental.datasets import UDPOS

        # smoke test to ensure imdb works properly
        train_dataset, valid_dataset, test_dataset = UDPOS()
        self.assertEqual(len(train_dataset), 12543)
        self.assertEqual(len(valid_dataset), 2002)
        self.assertEqual(len(test_dataset), 2077)
        self.assertEqual(train_dataset[0][0][:10],
                         torch.tensor([262, 16, 5728, 45, 289, 701, 1160, 4436, 10660, 585]).long())
        self.assertEqual(train_dataset[0][1][:10],
                         torch.tensor([8, 3, 8, 3, 9, 2, 4, 8, 8, 8]).long())
        self.assertEqual(train_dataset[0][2][:10],
                         torch.tensor([5, 34, 5, 27, 7, 11, 14, 5, 5, 5]).long())
        self.assertEqual(train_dataset[-1][0][:10],
                         torch.tensor([9, 32, 169, 436, 59, 192, 30, 6, 117, 17]).long())
        self.assertEqual(train_dataset[-1][1][:10],
                         torch.tensor([5, 10, 11, 4, 11, 11, 3, 12, 11, 4]).long())
        self.assertEqual(train_dataset[-1][2][:10],
                         torch.tensor([6, 20, 8, 10, 8, 8, 24, 13, 8, 15]).long())

        self.assertEqual(valid_dataset[0][0][:10],
                         torch.tensor([746, 3, 10633, 656, 25, 1334, 45]).long())
        self.assertEqual(valid_dataset[0][1][:10],
                         torch.tensor([6, 7, 8, 4, 7, 2, 3]).long())
        self.assertEqual(valid_dataset[0][2][:10],
                         torch.tensor([3, 4, 5, 16, 4, 2, 27]).long())
        self.assertEqual(valid_dataset[-1][0][:10],
                         torch.tensor([354, 4, 31, 17, 141, 421, 148, 6, 7, 78]).long())
        self.assertEqual(valid_dataset[-1][1][:10],
                         torch.tensor([11, 3, 5, 4, 9, 2, 2, 12, 7, 11]).long())
        self.assertEqual(valid_dataset[-1][2][:10],
                         torch.tensor([8, 12, 6, 15, 7, 2, 2, 13, 4, 8]).long())

        self.assertEqual(test_dataset[0][0][:10],
                         torch.tensor([210, 54, 3115, 0, 12229, 0, 33]).long())
        self.assertEqual(test_dataset[0][1][:10],
                         torch.tensor([5, 15, 8, 4, 6, 8, 3]).long())
        self.assertEqual(test_dataset[0][2][:10],
                         torch.tensor([30, 3, 5, 14, 3, 5, 9]).long())
        self.assertEqual(test_dataset[-1][0][:10],
                         torch.tensor([116, 0, 6, 11, 412, 10, 0, 4, 0, 6]).long())
        self.assertEqual(test_dataset[-1][1][:10],
                         torch.tensor([5, 4, 12, 10, 9, 15, 4, 3, 4, 12]).long())
        self.assertEqual(test_dataset[-1][2][:10],
                         torch.tensor([6, 16, 13, 16, 7, 3, 19, 12, 19, 13]).long())

        # Assert vocabs
        self.assertEqual(len(train_dataset.get_vocabs()), 3)
        self.assertEqual(len(train_dataset.get_vocabs()[0]), 19674)
        self.assertEqual(len(train_dataset.get_vocabs()[1]), 19)
        self.assertEqual(len(train_dataset.get_vocabs()[2]), 52)

        # Assert token ids
        word_vocab = train_dataset.get_vocabs()[0]
        tokens_ids = [word_vocab[token] for token in 'Two of them were being run'.split()]
        self.assertEqual(tokens_ids, [1206, 8, 69, 60, 157, 452])

    def test_conll_sequence_tagging(self):
        from torchtext.experimental.datasets import CoNLL2000Chunking

        # smoke test to ensure imdb works properly
        train_dataset, test_dataset = CoNLL2000Chunking()
        self.assertEqual(len(train_dataset), 8936)
        self.assertEqual(len(test_dataset), 2012)
        self.assertEqual(train_dataset[0][0][:10],
                         torch.tensor([11556, 9, 3, 1775, 17, 1164, 177, 6, 212, 317]).long())
        self.assertEqual(train_dataset[0][1][:10],
                         torch.tensor([2, 3, 5, 2, 17, 12, 16, 15, 13, 5]).long())
        self.assertEqual(train_dataset[0][2][:10],
                         torch.tensor([3, 6, 3, 2, 5, 7, 7, 7, 7, 3]).long())
        self.assertEqual(train_dataset[-1][0][:10],
                         torch.tensor([85, 17, 59, 6473, 288, 115, 72, 5, 2294, 2502]).long())
        self.assertEqual(train_dataset[-1][1][:10],
                         torch.tensor([18, 17, 12, 19, 10, 6, 3, 3, 4, 4]).long())
        self.assertEqual(train_dataset[-1][2][:10],
                         torch.tensor([3, 5, 7, 7, 3, 2, 6, 6, 3, 2]).long())

        self.assertEqual(test_dataset[0][0][:10],
                         torch.tensor([0, 294, 73, 10, 13582, 194, 18, 24, 2414, 7]).long())
        self.assertEqual(test_dataset[0][1][:10],
                         torch.tensor([4, 4, 4, 23, 4, 2, 11, 18, 11, 5]).long())
        self.assertEqual(test_dataset[0][2][:10],
                         torch.tensor([3, 2, 2, 3, 2, 2, 5, 3, 5, 3]).long())
        self.assertEqual(test_dataset[-1][0][:10],
                         torch.tensor([51, 456, 560, 2, 11, 465, 2, 1413, 36, 60]).long())
        self.assertEqual(test_dataset[-1][1][:10],
                         torch.tensor([3, 4, 4, 8, 3, 2, 8, 4, 17, 16]).long())
        self.assertEqual(test_dataset[-1][2][:10],
                         torch.tensor([6, 3, 2, 4, 6, 3, 4, 3, 5, 7]).long())

        # Assert vocabs
        self.assertEqual(len(train_dataset.get_vocabs()), 3)
        self.assertEqual(len(train_dataset.get_vocabs()[0]), 19124)
        self.assertEqual(len(train_dataset.get_vocabs()[1]), 46)
        self.assertEqual(len(train_dataset.get_vocabs()[2]), 24)

        # Assert token ids
        word_vocab = train_dataset.get_vocabs()[0]
        tokens_ids = [word_vocab[token] for token in 'Two of them were being run'.split()]
        self.assertEqual(tokens_ids, [970, 5, 135, 43, 214, 690])

    def test_squad1(self):
        from torchtext.experimental.datasets import SQuAD1
        from torchtext.vocab import Vocab
        # smoke test to ensure imdb works properly
        train_dataset, dev_dataset = SQuAD1()
        self.assertEqual(len(train_dataset), 87599)
        self.assertEqual(len(dev_dataset), 10570)
        context, question, answers, ans_pos = train_dataset[100]
        self.assertEqual(question,
                         torch.tensor([7, 24, 86, 52, 2, 373, 887, 18, 12797, 11090, 1356, 2, 1788, 3273, 16]).long())
        self.assertEqual(ans_pos[0], torch.tensor([72, 72]).long())
        context, question, answers, ans_pos = dev_dataset[100]
        self.assertEqual(question, torch.tensor([42, 27, 669, 7438, 17, 2, 1950, 3273, 17252, 389, 16]).long())
        self.assertEqual(ans_pos[0], torch.tensor([45, 48]).long())

        # Test API with a vocab input object
        old_vocab = train_dataset.get_vocab()
        new_vocab = Vocab(counter=old_vocab.freqs, max_size=2500)
        new_train_data, new_test_data = SQuAD1(vocab=new_vocab)

    def test_squad2(self):
        from torchtext.experimental.datasets import SQuAD2
        from torchtext.vocab import Vocab
        # smoke test to ensure imdb works properly
        train_dataset, dev_dataset = SQuAD2()
        self.assertEqual(len(train_dataset), 130319)
        self.assertEqual(len(dev_dataset), 11873)
        context, question, answers, ans_pos = train_dataset[200]
        self.assertEqual(question,
                         torch.tensor([84, 50, 1421, 12, 5439, 4569, 17, 30, 2, 15202, 4754, 1421, 16]).long())
        self.assertEqual(ans_pos[0], torch.tensor([9, 9]).long())
        context, question, answers, ans_pos = dev_dataset[200]
        self.assertEqual(question, torch.tensor([41, 29, 2, 66, 17016, 30, 0, 1955, 16]).long())
        self.assertEqual(ans_pos[0], torch.tensor([40, 46]).long())

        # Test API with a vocab input object
        old_vocab = train_dataset.get_vocab()
        new_vocab = Vocab(counter=old_vocab.freqs, max_size=2500)
        new_train_data, new_test_data = SQuAD2(vocab=new_vocab)
