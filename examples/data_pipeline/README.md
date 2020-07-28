# Data processing pipelines with torchtext

This example shows a few data processing pipeline with the building blocks (like tokenizer, vocab). The raw text data from `torchtext.experimental.datasets.raw.text_classification` are used as the inputs for performance benchmark. We also enable the JIT support if possible.


## SentencePiece 

This pipeline example shows the application with a pretrained sentencepiece model saved in `m_user.model`. The model is loaded to build tokenizer and vocabulary and the pipeline is composed of:

* `PretrainedSPTokenizer`
* `PretrainedSPVocab` backed by `torchtext.experimental.vocab.Vocab`
* `ToLongTensor` to convert a list of integers to `torch.tensor`

The command to run the pipeline:

    python pipelines.py --pipeline sentencepiece

The lookup time: 25.09393248707056 (eager mode) vs. 18.71099873096682 (jit mode)
