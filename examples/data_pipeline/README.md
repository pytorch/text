# Data processing pipelines with torchtext

This example shows a few data processing pipelines with the building blocks (like tokenizer, vocab). The raw text data from `torchtext.experimental.datasets.raw.text_classification` are used as the inputs for performance benchmark. We also enable the JIT support if possible.


## SentencePiece 

This pipeline example shows the application with a pretrained sentencepiece model saved in `m_user.model`. The model is loaded to build tokenizer and vocabulary and the pipeline is composed of:

* `PretrainedSPTokenizer`
* `PretrainedSPVocab` backed by `torchtext.experimental.vocab.Vocab`
* `ToLongTensor` to convert a list of integers to `torch.tensor`

The command to run the pipeline:

    python pipelines.py --pipeline sentencepiece

The lookup time: 25.09393248707056 (eager mode) vs. 18.71099873096682 (jit mode)


## Hugging Face 

This pipeline example shows the application with the vocab text file from Hugging Face ([link](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt)). The experimental vocab in torchtext library is used here:

* `torchtext.experimental.transforms.BasicEnglishNormalize` backed by `re2` regular expression library
* `torchtext.experimental.vocab.Vocab`
* `ToLongTensor` to convert a list of integers to `torch.tensor`

The command to run the pipeline:

    python pipelines.py --pipeline huggingface 

The lookup time: 45.7489246588666 (eager mode) vs. 30.990019195945933 (jit mode)


## PyText 

This pipeline example shows the application with the existing `ScriptVocab` in pytext library. The `ScriptVocab` instance is built from a text file where a column of vocab tokens are read in sequence.

* `torchtext.experimental.transforms.BasicEnglishNormalize` backed by `re2` regular expression library
* `from pytext.torchscript.vocab.ScriptVocabulary`
* `ToLongTensor` to convert a list of integers to `torch.tensor`

With the dependency of `pytext` library, the command to run the pipeline:

    python pipelines.py --pipeline pytext

The lookup time: 43.51164810406044 (eager mode) vs. 26.57643914804794 (jit mode)


## Torchtext

This pipeline example shows the application with the existing `Vocab` in torchtext library. The `Vocab` instance is built from a text file where a column of vocab tokens are read in sequence.

* `basic_english` func from `torchtext.data.utils.get_tokenizer`
* `torchtext.vocab.Vocab`
* `torchtext.experimental.functional.totensor` to convert a list of integers to `torch.tensor`

The command to run the pipeline:

    python pipelines.py --pipeline torchtext

The lookup time: 6.821685338858515 (eager mode)


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

The lookup time: 7.915330206044018 (eager mode)


## FastText pretrained word vectors 

This pipeline example shows the application with the pretained word vector from FastText:

* `torchtext.experimental.transforms.BasicEnglishNormalize` backed by `re2` regular expression library
* `torchtext.experimental.vectors.FastText`

The command to run the pipeline:

    python pipelines.py --pipeline fasttext 

The lookup time: 45.74515324481763 (eager mode) vs. 32.64012925699353 (jit mode)
