"""Run smoke tests"""

import torchtext
import torchtext.data # noqa: F401
import torchtext.data.batch # noqa: F401
import torchtext.data.dataset # noqa: F401
import torchtext.data.example # noqa: F401
import torchtext.data.field # noqa: F401
import torchtext.data.functional # noqa: F401
import torchtext.data.iterator # noqa: F401
import torchtext.data.metrics # noqa: F401
import torchtext.data.pipeline # noqa: F401
import torchtext.data.utils # noqa: F401
import torchtext.datasets # noqa: F401
import torchtext.datasets.babi # noqa: F401
import torchtext.datasets.imdb # noqa: F401
import torchtext.datasets.language_modeling # noqa: F401
import torchtext.datasets.nli # noqa: F401
import torchtext.datasets.sequence_tagging # noqa: F401
import torchtext.datasets.sst # noqa: F401
import torchtext.datasets.text_classification # noqa: F401
import torchtext.datasets.translation # noqa: F401
import torchtext.datasets.trec # noqa: F401
import torchtext.datasets.unsupervised_learning # noqa: F401
import torchtext.experimental # noqa: F401
import torchtext.experimental.datasets # noqa: F401
import torchtext.experimental.datasets.language_modeling # noqa: F401
import torchtext.experimental.datasets.text_classification # noqa: F401
import torchtext.utils # noqa: F401
import torchtext.vocab # noqa: F401

print('torchtext version is ', torchtext.__version__)
