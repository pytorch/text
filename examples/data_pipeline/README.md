# Data processing pipelines with torchtext

This example shows a few data processing pipelines with the building blocks (like tokenizer, vocab). The raw text data from `torchtext.experimental.datasets.raw.text_classification` are used as the inputs for performance benchmark. We also enable the JIT support if possible.


## SentencePiece 

This pipeline example shows the application with a pretrained sentencepiece model saved in `m_user.model`. The model is loaded to build tokenizer and vocabulary and the pipeline is composed of:

* `PretrainedSPTokenizer`
* `PretrainedSPVocab` backed by `torchtext.experimental.vocab.Vocab`
* `ToLongTensor` to convert a list of integers to `torch.tensor`

The command to run the pipeline:

    python pipelines.py --pipeline sentencepiece

The lookup time: 30.770548372063786 (eager mode with pybind)
The lookup time: 34.36592311505228 (eager mode with torchbind)
The lookup time: 23.43273439211771 (jit mode)


## Vocab from a text file 

This pipeline example shows the application with the vocab text file from Hugging Face ([link](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt)). The experimental vocab in torchtext library is used here:

* `torchtext.experimental.transforms.BasicEnglishNormalize` backed by `re2` regular expression library
* `torchtext.experimental.vocab.Vocab`
* `ToLongTensor` to convert a list of integers to `torch.tensor`

The command to run the pipeline:

    python pipelines.py --pipeline text_vocab 

The lookup time: 10.21763815311715 (eager mode with pybind)
The lookup time: 17.28485624492168 (eager mode with torchbind)
The lookup time: 10.25370063772425 (jit mode)


## PyText 

This pipeline example shows the application with the existing `ScriptVocab` in pytext library. The `ScriptVocab` instance is built from a text file where a column of vocab tokens are read in sequence.

* `torchtext.experimental.transforms.BasicEnglishNormalize` backed by `re2` regular expression library
* `from pytext.torchscript.vocab.ScriptVocabulary`
* `ToLongTensor` to convert a list of integers to `torch.tensor`

With the dependency of `pytext` library, the command to run the pipeline:

    python pipelines.py --pipeline pytext

The lookup time: 18.07144843228161 (eager mode with pybind)
The lookup time: 22.16066740499809 (eager mode with torchbind)
The lookup time: 13.41519635310396 (jit mode)


## Torchtext

This pipeline example shows the application with the existing `Vocab` in torchtext library. The `Vocab` instance is built from a text file where a column of vocab tokens are read in sequence.

* `basic_english` func from `torchtext.data.utils.get_tokenizer`
* `torchtext.vocab.Vocab`
* `torchtext.experimental.functional.totensor` to convert a list of integers to `torch.tensor`

The command to run the pipeline:

    python pipelines.py --pipeline torchtext

The lookup time: 8.690132656134665 (eager mode)


## Torchtext with a batch of data

This pipeline example shows the application with the data batch as input. For the real-world text classification task, two separate pipelines are created for text and label.

For the text pipeline:

* `basic_english` func from `torchtext.data.utils.get_tokenizer`
* `torchtext.vocab.Vocab`
* `torchtext.experimental.functional.totensor` to convert a list of integers to `torch.tensor`

For the label pipeline:

* `torchtext.experimental.functional.totensor` to convert a list of strings to `torch.tensor`

And the text and label pipeline are passed to TextClassificationPipeline. Since the incoming data are in the form of a batch, `run_batch_benchmark_lookup` func uses python built-in `map()` func to process a batch of raw text data according the pipeline.

The command to run the pipeline:

    python pipelines.py --pipeline batch_torchtext

The lookup time: 10.05315054487437 (eager mode)


## FastText pretrained word vectors 

This pipeline example shows the application with the pretained word vector from FastText:

* `torchtext.experimental.transforms.BasicEnglishNormalize` backed by `re2` regular expression library
* `torchtext.experimental.vectors.FastText`

The command to run the pipeline:

    python pipelines.py --pipeline fasttext 

The lookup time: 16.45024944096803 (eager mode with pybind)
The lookup time: 23.96459424262866 (eager mode with torchbind)
The lookup time: 19.34995342604816 (jit mode)
