#!/user/bin/env python3
# Note that all the tests in this module require dataset (either network access or cached)
import os
import torchtext.data as data
import torch
import torchtext
from ..common.torchtext_test_case import TorchtextTestCase
from ..common.assets import conditional_remove


class TestDataset(TorchtextTestCase):
    def _helper_test_func(self, length, target_length, results, target_results):
        self.assertEqual(length, target_length)
        if isinstance(target_results, list):
            target_results = torch.tensor(target_results, dtype=torch.int64)
        if isinstance(target_results, tuple):
            target_results = tuple(torch.tensor(item, dtype=torch.int64) for item in target_results)
        self.assertEqual(results, target_results)

    def test_wikitext2_legacy(self):
        from torchtext.datasets import WikiText2
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
        cachedir = os.path.join(self.project_root, ".data", "wikitext-2")
        conditional_remove(cachedir)
        cachefile = os.path.join(self.project_root, ".data", "wikitext-2-v1.zip")
        conditional_remove(cachefile)

        train_dataset, valid_dataset, test_dataset = WikiText2()
        self._helper_test_func(len(train_dataset), 2049990, train_dataset[20:25],
                               [5024, 89, 21, 3, 1838])
        self._helper_test_func(len(test_dataset), 241859, test_dataset[30:35],
                               [914, 4, 36, 11, 569])
        self._helper_test_func(len(valid_dataset), 214417, valid_dataset[40:45],
                               [925, 8, 2, 150, 8575])

        vocab = train_dataset.get_vocab()
        tokens_ids = [vocab[token] for token in 'the player characters rest'.split()]
        self.assertEqual(tokens_ids, [2, 286, 503, 700])

        # Add test for the subset of the standard datasets
        train_dataset, test_dataset = torchtext.experimental.datasets.raw.WikiText2(data_select=('train', 'test'))
        self._helper_test_func(len(train_dataset), 36718, next(iter(train_dataset)), ' \n')
        self._helper_test_func(len(test_dataset), 36718, next(iter(test_dataset)), ' \n')
        del train_dataset, test_dataset
        train_dataset, test_dataset = WikiText2(data_select=('train', 'test'))
        self._helper_test_func(len(train_dataset), 2049990, train_dataset[20:25],
                               [5024, 89, 21, 3, 1838])
        self._helper_test_func(len(test_dataset), 241859, test_dataset[30:35],
                               [914, 4, 36, 11, 569])

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
        self._helper_test_func(len(train_dataset), 924412, train_dataset[20:25],
                               [9919, 9920, 9921, 9922, 9188])
        self._helper_test_func(len(test_dataset), 82114, test_dataset[30:35],
                               [397, 93, 4, 16, 7])
        self._helper_test_func(len(valid_dataset), 73339, valid_dataset[40:45],
                               [0, 0, 78, 426, 196])

        vocab = train_dataset.get_vocab()
        tokens_ids = [vocab[token] for token in 'the player characters rest'.split()]
        self.assertEqual(tokens_ids, [2, 2550, 3344, 1125])

        # Add test for the subset of the standard datasets
        train_dataset, test_dataset = PennTreebank(data_select=('train', 'test'))
        self._helper_test_func(len(train_dataset), 924412, train_dataset[20:25],
                               [9919, 9920, 9921, 9922, 9188])
        self._helper_test_func(len(test_dataset), 82114, test_dataset[30:35],
                               [397, 93, 4, 16, 7])
        train_iter, test_iter = torchtext.experimental.datasets.raw.PennTreebank(data_select=('train', 'test'))
        self._helper_test_func(len(train_iter), 42068, next(iter(train_iter))[:15], ' aer banknote b')
        self._helper_test_func(len(test_iter), 42068, next(iter(test_iter))[:25], " no it was n't black mond")
        del train_dataset, test_dataset

    def test_text_classification(self):
        from torchtext.experimental.datasets import AG_NEWS
        # smoke test to ensure ag_news dataset works properly
        datadir = os.path.join(self.project_root, ".data")
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        train_dataset, test_dataset = AG_NEWS(root=datadir, ngrams=3)
        self._helper_test_func(len(train_dataset), 120000, train_dataset[-1][1][:10],
                               [3525, 319, 4053, 34, 5407, 3607, 70, 6798, 10599, 4053])
        self._helper_test_func(len(test_dataset), 7600, test_dataset[-1][1][:10],
                               [2351, 758, 96, 38581, 2351, 220, 5, 396, 3, 14786])

        # Add test for the subset of the standard datasets
        train_dataset, = AG_NEWS(data_select=('train'))
        self._helper_test_func(len(train_dataset), 120000, train_dataset[-1][1][:10],
                               [2155, 223, 2405, 30, 3010, 2204, 54, 3603, 4930, 2405])
        train_iter, = torchtext.experimental.datasets.raw.AG_NEWS(data_select=('train'))
        self._helper_test_func(len(train_iter), 120000, next(iter(train_iter))[1][:25], 'Wall St. Bears Claw Back ')
        del train_iter

    def test_imdb(self):
        from torchtext.experimental.datasets import IMDB
        from torchtext.vocab import Vocab
        # smoke test to ensure imdb works properly
        train_dataset, test_dataset = IMDB()
        self._helper_test_func(len(train_dataset), 25000, train_dataset[0][1][:10],
                               [13, 1568, 13, 246, 35468, 43, 64, 398, 1135, 92])
        self._helper_test_func(len(test_dataset), 25000, test_dataset[0][1][:10],
                               [13, 125, 1051, 5, 246, 1652, 8, 277, 66, 20])

        # Test API with a vocab input object
        old_vocab = train_dataset.get_vocab()
        new_vocab = Vocab(counter=old_vocab.freqs, max_size=2500)
        new_train_data, new_test_data = IMDB(vocab=new_vocab)

        # Add test for the subset of the standard datasets
        train_dataset, = IMDB(data_select=('train'))
        self._helper_test_func(len(train_dataset), 25000, train_dataset[0][1][:10],
                               [13, 1568, 13, 246, 35468, 43, 64, 398, 1135, 92])
        train_iter, = torchtext.experimental.datasets.raw.IMDB(data_select=('train'))
        self._helper_test_func(len(train_iter), 25000, next(iter(train_iter))[1][:25], 'I rented I AM CURIOUS-YEL')
        del train_dataset, test_dataset

    def test_multi30k(self):
        from torchtext.experimental.datasets import Multi30k
        # smoke test to ensure multi30k works properly
        train_dataset, valid_dataset, test_dataset = Multi30k()
        self._helper_test_func(len(train_dataset), 29000, train_dataset[20],
                               ([3, 443, 2530, 46, 17478, 7422, 7, 157, 9, 11, 5848, 2],
                                [4, 60, 529, 136, 1493, 9, 8, 279, 5, 2, 3748, 3]))
        self._helper_test_func(len(valid_dataset), 1014, valid_dataset[30],
                               ([3, 178, 25, 84, 1003, 56, 18, 153, 2],
                                [4, 23, 31, 80, 46, 1347, 5, 2, 118, 3]))
        self._helper_test_func(len(test_dataset), 1000, test_dataset[40],
                               ([3, 25, 5, 11, 3914, 1536, 20, 63, 2],
                                [4, 31, 19, 2, 746, 344, 1914, 5, 45, 3]))

        de_vocab, en_vocab = train_dataset.get_vocab()
        de_tokens_ids = [
            de_vocab[token] for token in
            'Zwei Männer verpacken Donuts in Kunststofffolie'.split()
        ]
        self.assertEqual(de_tokens_ids, [19, 29, 18703, 4448, 5, 6240])

        en_tokens_ids = [
            en_vocab[token] for token in
            'Two young White males are outside near many bushes'.split()
        ]
        self.assertEqual(en_tokens_ids,
                         [17, 23, 1167, 806, 15, 55, 82, 334, 1337])

        # Add test for the subset of the standard datasets
        train_dataset, = torchtext.experimental.datasets.raw.Multi30k(data_select=('train'))
        self._helper_test_func(len(train_dataset), 29000, ' '.join(next(iter(train_dataset))),
                               ' '.join(['Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.',
                                         'Two young  White males are outside near many bushes.']))
        del train_dataset, test_dataset
        train_dataset, = Multi30k(data_select=('train'))
        self._helper_test_func(len(train_dataset), 29000, train_dataset[20],
                               ([3, 443, 2530, 46, 17478, 7422, 7, 157, 9, 11, 5848, 2],
                                [4, 60, 529, 136, 1493, 9, 8, 279, 5, 2, 3748, 3]))

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
        self._helper_test_func(len(train_dataset), 12543, (train_dataset[0][0][:10], train_dataset[0][1][:10],
                                                           train_dataset[0][2][:10], train_dataset[-1][0][:10],
                                                           train_dataset[-1][1][:10], train_dataset[-1][2][:10]),
                               ([262, 16, 5728, 45, 289, 701, 1160, 4436, 10660, 585],
                                [8, 3, 8, 3, 9, 2, 4, 8, 8, 8],
                                [5, 34, 5, 27, 7, 11, 14, 5, 5, 5],
                                [9, 32, 169, 436, 59, 192, 30, 6, 117, 17],
                                [5, 10, 11, 4, 11, 11, 3, 12, 11, 4],
                                [6, 20, 8, 10, 8, 8, 24, 13, 8, 15]))
        self._helper_test_func(len(valid_dataset), 2002, (valid_dataset[0][0][:10], valid_dataset[0][1][:10],
                                                          valid_dataset[0][2][:10], valid_dataset[-1][0][:10],
                                                          valid_dataset[-1][1][:10], valid_dataset[-1][2][:10]),
                               ([746, 3, 10633, 656, 25, 1334, 45],
                                [6, 7, 8, 4, 7, 2, 3],
                                [3, 4, 5, 16, 4, 2, 27],
                                [354, 4, 31, 17, 141, 421, 148, 6, 7, 78],
                                [11, 3, 5, 4, 9, 2, 2, 12, 7, 11],
                                [8, 12, 6, 15, 7, 2, 2, 13, 4, 8]))
        self._helper_test_func(len(test_dataset), 2077, (test_dataset[0][0][:10], test_dataset[0][1][:10],
                                                         test_dataset[0][2][:10], test_dataset[-1][0][:10],
                                                         test_dataset[-1][1][:10], test_dataset[-1][2][:10]),
                               ([210, 54, 3115, 0, 12229, 0, 33],
                                [5, 15, 8, 4, 6, 8, 3],
                                [30, 3, 5, 14, 3, 5, 9],
                                [116, 0, 6, 11, 412, 10, 0, 4, 0, 6],
                                [5, 4, 12, 10, 9, 15, 4, 3, 4, 12],
                                [6, 16, 13, 16, 7, 3, 19, 12, 19, 13]))

        # Assert vocabs
        self.assertEqual(len(train_dataset.get_vocabs()), 3)
        self.assertEqual(len(train_dataset.get_vocabs()[0]), 19674)
        self.assertEqual(len(train_dataset.get_vocabs()[1]), 19)
        self.assertEqual(len(train_dataset.get_vocabs()[2]), 52)

        # Assert token ids
        word_vocab = train_dataset.get_vocabs()[0]
        tokens_ids = [word_vocab[token] for token in 'Two of them were being run'.split()]
        self.assertEqual(tokens_ids, [1206, 8, 69, 60, 157, 452])

        # Add test for the subset of the standard datasets
        train_dataset, = UDPOS(data_select=('train'))
        self._helper_test_func(len(train_dataset), 12543, (train_dataset[0][0][:10], train_dataset[-1][2][:10]),
                               ([262, 16, 5728, 45, 289, 701, 1160, 4436, 10660, 585],
                                [6, 20, 8, 10, 8, 8, 24, 13, 8, 15]))
        train_iter, = torchtext.experimental.datasets.raw.UDPOS(data_select=('train'))
        self._helper_test_func(len(train_iter), 12543, ' '.join(next(iter(train_iter))[0][:5]),
                               ' '.join(['Al', '-', 'Zaman', ':', 'American']))
        del train_iter

    def test_conll_sequence_tagging(self):
        from torchtext.experimental.datasets import CoNLL2000Chunking

        # smoke test to ensure imdb works properly
        train_dataset, test_dataset = CoNLL2000Chunking()
        self._helper_test_func(len(train_dataset), 8936, (train_dataset[0][0][:10], train_dataset[0][1][:10],
                                                          train_dataset[0][2][:10], train_dataset[-1][0][:10],
                                                          train_dataset[-1][1][:10], train_dataset[-1][2][:10]),
                               ([11556, 9, 3, 1775, 17, 1164, 177, 6, 212, 317],
                                [2, 3, 5, 2, 17, 12, 16, 15, 13, 5],
                                [3, 6, 3, 2, 5, 7, 7, 7, 7, 3],
                                [85, 17, 59, 6473, 288, 115, 72, 5, 2294, 2502],
                                [18, 17, 12, 19, 10, 6, 3, 3, 4, 4],
                                [3, 5, 7, 7, 3, 2, 6, 6, 3, 2]))
        self._helper_test_func(len(test_dataset), 2012, (test_dataset[0][0][:10], test_dataset[0][1][:10],
                                                         test_dataset[0][2][:10], test_dataset[-1][0][:10],
                                                         test_dataset[-1][1][:10], test_dataset[-1][2][:10]),
                               ([0, 294, 73, 10, 13582, 194, 18, 24, 2414, 7],
                                [4, 4, 4, 23, 4, 2, 11, 18, 11, 5],
                                [3, 2, 2, 3, 2, 2, 5, 3, 5, 3],
                                [51, 456, 560, 2, 11, 465, 2, 1413, 36, 60],
                                [3, 4, 4, 8, 3, 2, 8, 4, 17, 16],
                                [6, 3, 2, 4, 6, 3, 4, 3, 5, 7]))

        # Assert vocabs
        self.assertEqual(len(train_dataset.get_vocabs()), 3)
        self.assertEqual(len(train_dataset.get_vocabs()[0]), 19124)
        self.assertEqual(len(train_dataset.get_vocabs()[1]), 46)
        self.assertEqual(len(train_dataset.get_vocabs()[2]), 24)

        # Assert token ids
        word_vocab = train_dataset.get_vocabs()[0]
        tokens_ids = [word_vocab[token] for token in 'Two of them were being run'.split()]
        self.assertEqual(tokens_ids, [970, 5, 135, 43, 214, 690])

        # Add test for the subset of the standard datasets
        train_dataset, = CoNLL2000Chunking(data_select=('train'))
        self._helper_test_func(len(train_dataset), 8936, (train_dataset[0][0][:10], train_dataset[0][1][:10],
                                                          train_dataset[0][2][:10], train_dataset[-1][0][:10],
                                                          train_dataset[-1][1][:10], train_dataset[-1][2][:10]),
                               ([11556, 9, 3, 1775, 17, 1164, 177, 6, 212, 317],
                                [2, 3, 5, 2, 17, 12, 16, 15, 13, 5],
                                [3, 6, 3, 2, 5, 7, 7, 7, 7, 3],
                                [85, 17, 59, 6473, 288, 115, 72, 5, 2294, 2502],
                                [18, 17, 12, 19, 10, 6, 3, 3, 4, 4],
                                [3, 5, 7, 7, 3, 2, 6, 6, 3, 2]))
        train_iter, = torchtext.experimental.datasets.raw.CoNLL2000Chunking(data_select=('train'))
        self._helper_test_func(len(train_iter), 8936, ' '.join(next(iter(train_iter))[0][:5]),
                               ' '.join(['Confidence', 'in', 'the', 'pound', 'is']))
        del train_iter

    def test_squad1(self):
        from torchtext.experimental.datasets import SQuAD1
        from torchtext.vocab import Vocab
        # smoke test to ensure imdb works properly
        train_dataset, dev_dataset = SQuAD1()
        context, question, answers, ans_pos = train_dataset[100]
        self._helper_test_func(len(train_dataset), 87599, (question[:5], ans_pos[0]),
                               ([7, 24, 86, 52, 2], [72, 72]))
        context, question, answers, ans_pos = dev_dataset[100]
        self._helper_test_func(len(dev_dataset), 10570, (question, ans_pos[0]),
                               ([42, 27, 669, 7438, 17, 2, 1950, 3273, 17252, 389, 16], [45, 48]))

        # Test API with a vocab input object
        old_vocab = train_dataset.get_vocab()
        new_vocab = Vocab(counter=old_vocab.freqs, max_size=2500)
        new_train_data, new_test_data = SQuAD1(vocab=new_vocab)

        # Add test for the subset of the standard datasets
        train_dataset, = SQuAD1(data_select=('train'))
        context, question, answers, ans_pos = train_dataset[100]
        self._helper_test_func(len(train_dataset), 87599, (question[:5], ans_pos[0]),
                               ([7, 24, 86, 52, 2], [72, 72]))
        train_iter, = torchtext.experimental.datasets.raw.SQuAD1(data_select=('train'))
        self._helper_test_func(len(train_iter), 87599, next(iter(train_iter))[0][:50],
                               'Architecturally, the school has a Catholic charact')
        del train_iter

    def test_squad2(self):
        from torchtext.experimental.datasets import SQuAD2
        from torchtext.vocab import Vocab
        # smoke test to ensure imdb works properly
        train_dataset, dev_dataset = SQuAD2()
        context, question, answers, ans_pos = train_dataset[200]
        self._helper_test_func(len(train_dataset), 130319, (question[:5], ans_pos[0]),
                               ([84, 50, 1421, 12, 5439], [9, 9]))
        context, question, answers, ans_pos = dev_dataset[200]
        self._helper_test_func(len(dev_dataset), 11873, (question, ans_pos[0]),
                               ([41, 29, 2, 66, 17016, 30, 0, 1955, 16], [40, 46]))

        # Test API with a vocab input object
        old_vocab = train_dataset.get_vocab()
        new_vocab = Vocab(counter=old_vocab.freqs, max_size=2500)
        new_train_data, new_test_data = SQuAD2(vocab=new_vocab)

        # Add test for the subset of the standard datasets
        train_dataset, = SQuAD2(data_select=('train'))
        context, question, answers, ans_pos = train_dataset[200]
        self._helper_test_func(len(train_dataset), 130319, (question[:5], ans_pos[0]),
                               ([84, 50, 1421, 12, 5439], [9, 9]))
        train_iter, = torchtext.experimental.datasets.raw.SQuAD2(data_select=('train'))
        self._helper_test_func(len(train_iter), 130319, next(iter(train_iter))[0][:50],
                               'Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-Y')
        del train_iter
