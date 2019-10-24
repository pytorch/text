from torchtext.data import metrics
from .common.torchtext_test_case import TorchtextTestCase


class TestUtils(TorchtextTestCase):

    def test_bleu_score(self):
        # Full match
        candidate = [['My', 'full', 'pytorch', 'test']]
        refs = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']]]
        assert metrics.bleu_score(candidate, refs) == 1

        # No 4-gram
        candidate = [['My', 'full', 'pytorch']]
        refs = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']]]
        assert metrics.bleu_score(candidate, refs) == 0

        # Partial match
        candidate = [['My', 'full', 'pytorch', 'test']]
        refs = [[['My', 'full', 'pytorch', 'test', '!'], ['Different']]]
        assert round(metrics.bleu_score(candidate, refs), 4) == 0.7788

        # Bigrams and unigrams only
        candidate = [['My', 'pytorch', 'test']]
        refs = [[['My', 'full', 'pytorch', 'test'], ['Different']]]
        assert round(metrics.bleu_score(candidate, refs, max_n=2,
                     weights=[0.5, 0.5]), 4) == 0.5067

        # Multi-sentence corpus
        candidate = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
        refs = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']],
                [['No', 'Match']]]
        assert round(metrics.bleu_score(candidate, refs), 4) == 0.8409
