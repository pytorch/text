from torchtext.data import metrics
from torch.testing import assert_allclose
from ..common.torchtext_test_case import TorchtextTestCase


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
        assert_allclose(bleu_score(candidate, refs), 0.7788)

        # Bigrams and unigrams only
        candidate = [['My', 'pytorch', 'test']]
        refs = [[['My', 'full', 'pytorch', 'test'], ['Different']]]
        assert_allclose(bleu_score(candidate, refs, max_n=2,
                                   weights=[0.5, 0.5]), 0.5067)

        # Multi-sentence corpus
        candidate = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
        refs = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']],
                [['No', 'Match']]]
        assert_allclose(bleu_score(candidate, refs), 0.8409)

        # Empty input
        candidate = [[]]
        refs = [[[]]]
        assert metrics.bleu_score(candidate, refs) == 0

        # Long input, compared to NLTK implementation score
        candidate = [['Lucille', 'Bluth', 'has', '3', 'sons'],
                     ['She', 'loves', 'all', 'her', 'children', 'equally'],
                     ['No', 'match', 'here', 'at', 'all']]

        refs = [[['I', 'heard', 'Lucille', 'has', 'three', 'sons'],
                ['Rumor', 'has', 'it', 'Lucille', 'has', '3', 'sons', '!']],
                [['I', 'love', 'all', 'my', 'children', 'equally'],
                ['She', 'loves', 'all', 'her', 'children', 'equally']],
                [['I', 'have', 'made', 'a', 'terrible', 'mistake'], ['Big', 'mistake']]]

        # Value computed using nltk.translate.bleu_score.corpus_bleu(refs, candidate)
        assert_allclose(bleu_score(candidate, refs), 0.4573)
        assert_allclose(bleu_score(candidate, refs, 3,
                     weights=[0.33, 0.33, 0.33]), 0.4901)
        assert_allclose(bleu_score(candidate, refs, 2,
                     weights=[0.5, 0.5]), 0.5120)
        assert_allclose(bleu_score(candidate, refs, 1,
                     weights=[1]), 0.5516)
