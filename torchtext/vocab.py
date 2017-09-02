import array
from collections import defaultdict
import logging
import os
import zipfile

import six
from six.moves.urllib.request import urlretrieve
import torch
from tqdm import tqdm
import tarfile

from .utils import reporthook

logger = logging.getLogger(__name__)


class Vocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.

    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>'],
                 vectors=None, unk_init=torch.Tensor.zero_, expand_vocab=False):
        """Create a Vocab object from a collections.Counter.

        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vectors: one of the available pretrained vectors or a list with each
                element one of the available pretrained vectors
                (see Vocab.load_vectors). Default: None
            unk_init (callback): by default, initalize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size
            expand_vocab (bool): If True, expand vocabulary to include all
                words for which the specified pretrained word vectors are
                available. Default: False
        """
        self.freqs = counter.copy()
        min_freq = max(min_freq, 1)
        counter.update(['<unk>'] + specials)

        self.stoi = defaultdict(lambda: 0)
        self.stoi.update({tok: i + 1 for i, tok in enumerate(specials)})
        self.itos = ['<unk>'] + specials

        counter.subtract({tok: counter[tok] for tok in ['<unk>'] + specials})
        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
            self.stoi[word] = len(self.itos) - 1

        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, expand_vocab=expand_vocab)

    def __len__(self):
        return len(self.itos)

    def load_vectors(self, vectors, unk_init=torch.Tensor.zero_, expand_vocab=False):
        """Arguments:
              vectors: one of the available pretrained vectors or a list with each
                  element one of the available pretrained vectors:
                       glove.42B.300d
                       glove.840B.300d
                       glove.twitter.27B.25d
                       glove.twitter.27B.50d
                       glove.twitter.27B.100d
                       glove.twitter.27B.200d
                       glove.6B.50d
                       glove.6B.100d
                       glove.6B.200d
                       glove.6B.300d
                       charngram.100d
              unk_init (callback): by default, initalize out-of-vocabulary word vectors
                  to zero vectors; can be any function that takes in a Tensor and
                  returns a Tensor of the same size
              expand_vocab (bool): expand vocabulary to include all words for which
                  the specified pretrained word vectors are available
        """
        if not isinstance(vectors, list):
            vectors = [vectors]
        vecs = []
        tot_dim = 0
        for v in vectors:
            wv_type, _, rest = v.partition('.')
            rest, _, wv_dim = rest.rpartition('.')
            wv_dim = int(wv_dim[:-1])
            if wv_type == 'glove':
                wv_name = rest
                vecs.append(GloVe(name=wv_name, dim=wv_dim, unk_init=unk_init))
                if expand_vocab:
                    for w in sorted(vecs[-1].stoi.keys()):
                        self.itos.append(w)
                        self.stoi[w] = len(self.itos) - 1
            elif 'charngram' in v:
                vecs.append(CharNGram(unk_init=unk_init))
            tot_dim += wv_dim

        self.vectors = torch.Tensor(len(self), tot_dim)
        for i, token in enumerate(self.itos):
            start_dim = 0
            for j, v in enumerate(vectors):
                end_dim = start_dim + vecs[j].dim
                self.vectors[i][start_dim:end_dim] = vecs[j][token]
                start_dim = end_dim
            assert(start_dim == tot_dim)

    def set_vectors(self, stoi, vectors, dim, unk_init=torch.Tensor.zero_):
        self.vectors = torch.Tensor(len(self), dim)
        for i, token in enumerate(self.itos):
            wv_index = stoi.get(token, None)
            if wv_index is None:
                self.vectors[i] = vectors[wv_index]
            else:
                self.vectors[i] = unk_init(self.vectors[i])


class Vectors(object):

    def __init__(self, unk_init=torch.Tensor.zero_):
        self.unk_init = unk_init

    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            return self.unk_init(torch.Tensor(1, self.dim))

    def vector_cache(self, url, root, fname):
        desc = fname
        fname = os.path.join(root, fname)
        fname_pt = fname + '.pt'
        fname_txt = fname + '.txt'
        desc = os.path.basename(fname)

        if not os.path.isfile(fname_pt):
            dest = os.path.join(root, os.path.basename(url))
            if not os.path.isfile(fname_txt):
                logger.info('Downloading vectors from {}'.format(url))
                if not os.path.exists(root):
                    os.makedirs(root)
                with tqdm(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
                    urlretrieve(url, dest, reporthook=reporthook(t))
                logger.info('Extracting vectors into {}'.format(root))
                ext = os.path.splitext(dest)[1][1:]
                if ext == 'zip':
                    with zipfile.ZipFile(dest, "r") as zf:
                        zf.extractall(root)
                elif ext == 'gz':
                    with tarfile.open(dest, 'r:gz') as tar:
                        tar.extractall(path=root)
                else:
                    raise RuntimeError('unsupported compression format')
            if not os.path.isfile(fname_txt):
                raise RuntimeError('no vectors found')

            itos, vectors, dim = [], array.array('d'), None
            with open(fname_txt, 'rb') as f:
                lines = [line for line in f]
            logger.info("Loading vectors from {}".format(fname_txt))
            for line in tqdm(lines, total=len(lines)):
                entries = line.strip().split(b' ')
                word, entries = entries[0], entries[1:]
                if dim is None:
                    dim = len(entries)
                elif dim != len(entries):
                    raise RuntimeError(
                        "Vector for token {} has {} dimensions, but previously "
                        "read vectors have {} dimensions. All vectors must have "
                        "the same number of dimensions".format(word, len(entries), dim))
                try:
                    if isinstance(word, six.binary_type):
                        word = word.decode('utf-8')
                except:
                    logger.info('non-UTF8 token', repr(word), 'ignored')
                    continue
                vectors.extend(float(x) for x in entries)
                itos.append(word)

            self.stoi = {word: i for i, word in enumerate(itos)}
            self.vectors = torch.Tensor(vectors).view(-1, dim)
            self.dim = dim
            logger.info('saving vectors to', fname_pt)
            torch.save((self.stoi, self.vectors, self.dim), fname_pt)
        else:
            logger.info('loading vectors from', fname_pt)
            self.stoi, self.vectors, self.dim = torch.load(fname_pt)


class GloVe(Vectors):

    url = {
        'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
        'glove.test_twitter.27B': None,
    }

    def __init__(self, root='.vector_cache', name='840B', dim=300, **kwargs):
        super(GloVe, self).__init__(**kwargs)
        dim = str(dim) + 'd'
        name = '.'.join(['glove', name])
        fname = name + '.' + dim
        self.vector_cache(self.url[name], root, fname)


class CharNGram(Vectors):

    url = ('http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/'
           'jmt_pre-trained_embeddings.tar.gz')
    filename = 'charNgram'

    def __init__(self, root='.vector_cache', **kwargs):
        super(CharNGram, self).__init__(**kwargs)
        self.vector_cache(self.url, root, self.filename)

    def __getitem__(self, token):
        chars = ['#BEGIN#'] + list(token) + ['#END#']
        vector = torch.Tensor(1, self.dim).zero_()
        num_vectors = 0
        for n in [2, 3, 4]:
            end = len(chars) - n + 1
            grams = [chars[i:(i + n)] for i in range(end)]
            for gram in grams:
                gram_key = '{}gram-{}'.format(n, ''.join(gram))
                if gram_key in self.stoi:
                    vector += self.vectors[self.stoi[gram_key]]
        if num_vectors > 0:
            vector /= num_vectors
        else:
            vector = self.unk_init(vector)
        return vector
