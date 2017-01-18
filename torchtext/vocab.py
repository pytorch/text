import array
from collections import Counter
from collections import defaultdict
import os
import zipfile

import torch

def load_word_vectors(path):

    if os.path.isfile(path + '.pt'):
        print('loading word vectors from', path + '.pt')
        return torch.load(path + '.pt')

    if os.path.isfile(path + '.txt'):
        print('loading word vectors from', path + '.txt')
        cm = open(path + '.txt', 'rb')
    else:
        print('loading word vectors from', path + '.zip')
        with zipfile.ZipFile(path + '.zip') as zf:
            cm = zf.open(zf.infolist()[0].filename, 'rb')

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    with cm as f:
        for line in f:
            entries = line.strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                word = word.decode()
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_arr, wv_size)
    torch.save(ret, path + '.pt')
    return ret


class Vocab:

    def __init__(self, counter, max_size=None, min_freq=1, wv_path=None,
                 lower=False, specials=['<pad>'], fill_from_vectors=False):
        # self.specials = specials
        self.freqs = counter.copy()
        counter.update(['<unk>'] + specials)

        if wv_path is not None:
            wv_dict, wv_arr, self.wv_size = load_word_vectors(wv_path)

            if fill_from_vectors:
                counter.update(wv_dict.keys())

        self.stoi = defaultdict(lambda: 0)
        self.stoi.update({tok: i + 1 for i, tok in enumerate(specials)})
        self.itos = ['<unk>'] + specials

        counter.subtract({tok: counter[tok] for tok in ['<unk>'] + specials})
        max_size = None if max_size is None else max_size - len(self.itos)

        # sort by frequency, then alphabetically
        words = sorted(counter.items(), key=lambda tup: tup[0])
        words.sort(key=lambda tup: tup[1], reverse=True)

        for k, v in words:
            if v < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(k)
            self.stoi[k] = len(self.itos) - 1

        class LowercaseDict(defaultdict):

            @staticmethod
            def default_factory():
                return 0

            def __getitem__(self, key):
                return super().__getitem__(key.lower())

        if lower:
            self.stoi = LowercaseDict(self.stoi)
            self.freqs = LowercaseDict(self.freqs)

        if wv_path is not None:
            # this should be parametric (zeros or random)
            self.vectors = torch.zeros(len(self), self.wv_size)
            for i, token in enumerate(self.itos):
                wv_index = wv_dict.get(token, None)
                if wv_index is not None:
                    self.vectors[i] = wv_arr[wv_index]

    def __len__(self):
        return len(self.itos)
