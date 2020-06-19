from torchtext.data.metrics import bleu_score
from ..common.torchtext_test_case import TorchtextTestCase


class TestUtils(TorchtextTestCase):

    def test_bleu_score(self):
        # Full match
        candidate = [['My', 'full', 'pytorch', 'test']]
        refs = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']]]
        assert bleu_score(candidate, refs) == 1

        # No 4-gram
        candidate = [['My', 'full', 'pytorch']]
        refs = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']]]
        assert bleu_score(candidate, refs) == 0

        # Partial match
        candidate = [['My', 'full', 'pytorch', 'test']]
        refs = [[['My', 'full', 'pytorch', 'test', '!'], ['Different']]]
        self.assertEqual(bleu_score(candidate, refs), 0.7788007)

        # Bigrams and unigrams only
        candidate = [['My', 'pytorch', 'test']]
        refs = [[['My', 'full', 'pytorch', 'test'], ['Different']]]
        self.assertEqual(bleu_score(candidate, refs, max_n=2,
                                    weights=[0.5, 0.5]), 0.5066641)

        # Multi-sentence corpus
        candidate = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
        refs = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']],
                [['No', 'Match']]]
        self.assertEqual(bleu_score(candidate, refs), 0.8408964)

        # Empty input
        candidate = [[]]
        refs = [[[]]]
        assert bleu_score(candidate, refs) == 0

        # Long input, compared to NLTK implementation score
        # nltl version used: 3.4.5
        candidate = [['Lucille', 'B', 'has', '3', 'sons'],
                     ['She', 'loves', 'all', 'her', 'children', 'equally'],
                     ['No', 'match', 'here', 'at', 'all']]

        refs = [[['I', 'heard', 'Lucille', 'has', 'three', 'sons'],
                ['Rumor', 'has', 'it', 'Lucille', 'has', '3', 'sons', '!']],
                [['I', 'love', 'all', 'my', 'children', 'equally'],
                ['She', 'loves', 'all', 'her', 'children', 'equally']],
                [['I', 'have', 'made', 'a', 'terrible', 'mistake'], ['Big', 'mistake']]]

        # The comments below give the code used to get each hardcoded bleu score
        # nltk.translate.bleu_score.corpus_bleu(refs, candidate)
        self.assertEqual(bleu_score(candidate, refs), 0.4573199)
        # nltk.translate.bleu_score.corpus_bleu(refs, candidate, weights=[0.33]*3)
        self.assertEqual(bleu_score(candidate, refs, 3,
                         weights=[0.33, 0.33, 0.33]), 0.4901113)
        # nltk.translate.bleu_score.corpus_bleu(refs, candidate, weights=[0.5]*2)
        self.assertEqual(bleu_score(candidate, refs, 2,
                         weights=[0.5, 0.5]), 0.5119535)
        # nltk.translate.bleu_score.corpus_bleu(refs, candidate, weights=[1])
        self.assertEqual(bleu_score(candidate, refs, 1,
                         weights=[1]), 0.5515605)
