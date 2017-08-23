import numpy as np
import unittest

from torchtext import vocab
from collections import Counter


class TestVocab(unittest.TestCase):
    def test_vocab(self):
        c = Counter(['hello', 'world'])
        v = vocab.Vocab(c, vectors=['glove.twitter.27B.200d', 'charngram.100d'])

        self.assertEqual(v.itos, ['<unk>', '<pad>', 'hello', 'world'])
        vectors = v.vectors.numpy()

        # The first 5 entries in each vector.
        expected_glove_twitter = {
            'hello': [0.34683, -0.19612, -0.34923, -0.28158, -0.75627],
            'world': [0.035771, 0.62946, 0.27443, -0.36455, 0.39189],
        }

        for word in ['hello', 'world']:
            self.assertTrue(
                np.allclose(
                    vectors[v.stoi[word], :5], expected_glove_twitter[word]
                )
            )

        self.assertTrue(np.allclose(vectors[v.stoi['<unk>'], :], np.zeros(300)))


if __name__ == '__main__':
    unittest.main()
