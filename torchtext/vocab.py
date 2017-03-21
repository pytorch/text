from __future__ import print_function
import array
from collections import defaultdict
import os
import zipfile
import requests
import six

import torch


URL = {
        'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
        }

def return_zero():
    return 0

def return_first(x):
    return x[0]

def return_second(x):
    return x[1]

def load_word_vectors(root, wv_type, wv_dim):
    """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
    dim = str(wv_dim) + 'd'
    fname = os.path.join(root, wv_type + '.' + dim)
    if os.path.isfile(fname + '.pt'):
        fname_pt = fname + '.pt'
        print('loading word vectors from', fname_pt)
        return torch.load(fname_pt)
    if os.path.isfile(fname + '.txt'):
        fname_txt = fname + '.txt'
        print('loading word vectors from', fname_txt)
        cm = open(fname_txt, 'rb')
    elif os.path.basename(wv_type) in URL:
        url = URL[wv_type]
        print('downloading word vectors from {}'.format(url))
        r = requests.get(url, stream=True)
        with zipfile.ZipFile(six.BytesIO(r.content)) as zf:
            print('extracting word vectors into {}'.format(root))
            zf.extractall(root)
        return load_word_vectors(root, wv_type, wv_dim)
    else:
        print('Unable to load word vectors.')

    wv_tokens, wv_arr = [], array.array('d')
    with cm as f:
        for line in f:
            entries = line.strip().split(b' ')
            word, entries = entries[0], entries[1:]
            try:
                word = word.decode()
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_dim)
    ret = (wv_dict, wv_arr)
    torch.save(ret, fname + '.pt')
    return ret


class Vocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.

    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
        vectors: A Tensor containing word vectors for the tokens in the Vocab,
            if a word vector file has been provided.
    """

    def __init__(self, counter, max_size=None, min_freq=1, wv_dir=os.getcwd(),
                 wv_type=None, wv_dim=300, unk_init='random',
                 specials=['<pad>'], fill_from_vectors=False):
        """Create a Vocab object from a collections.Counter.

        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Default: 1.
            wv_dir: directory containing word vector file and destination for
                downloaded word vector files
            wv_type: type of word vectors; None for no word vectors
            wv_dim: dimension of word vectors
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token.
            fill_from_vectors: Whether to add to the vocabulary every token
                for which a word vector specified by vectors is present
                even if the token does not appear in the provided data.
            unk_init: default to random initialization for word vectors not in the
                pretrained word vector file; otherwise set to zero
        """
        self.freqs = counter.copy()
        self.unk_init = unk_init
        counter.update(['<unk>'] + specials)

        if wv_type:
            os.makedirs(wv_dir, exist_ok=True)
            wv_dict, wv_arr = load_word_vectors(wv_dir, wv_type, wv_dim)

            if fill_from_vectors:
                counter.update(wv_dict.keys())

        self.stoi = defaultdict(return_zero)
        self.stoi.update({tok: i + 1 for i, tok in enumerate(specials)})
        self.itos = ['<unk>'] + specials

        counter.subtract({tok: counter[tok] for tok in ['<unk>'] + specials})
        max_size = None if max_size is None else max_size - len(self.itos)

        # sort by frequency, then alphabetically
        words = sorted(counter.items(), key=return_first)
        words.sort(key=return_second, reverse=True)

        for k, v in words:
            if v < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(k)
            self.stoi[k] = len(self.itos) - 1

        if wv_type:
            self.set_vectors(wv_dict, wv_arr, wv_dim)

    def __len__(self):
        return len(self.itos)

    def load_vectors(self, wv_dir=os.getcwd(), wv_type=None, wv_dim=300, unk_init='random'):
        """Loads word vectors into the vocab

        Arguments:
            wv_dir: directory containing word vector file and destination for
                downloaded word vector files
            wv_type: type of word vectors; None for no word vectors
            wv_dim: dimension of word vectors
            unk_init: default to random initialization for unknown word vectors;
                otherwise set to zero
        """
        self.unk_init = unk_init
        wv_dict, wv_arr = load_word_vectors(wv_dir, wv_type, wv_dim)
        self.set_vectors(wv_dict, wv_arr, wv_dim)

    def set_vectors(self, wv_dict, wv_arr, wv_dim):
        self.vectors = torch.Tensor(len(self), wv_dim)
        self.vectors.normal_(0, 1) if self.unk_init == 'random' else self.vectors.zero_()
        for i, token in enumerate(self.itos):
            wv_index = wv_dict.get(token, None)
            if wv_index is not None:
                self.vectors[i] = wv_arr[wv_index]
