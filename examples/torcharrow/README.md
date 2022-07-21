## Description

This example shows end-2-end training for SST-2 binary classification using the RoBERTa model and TorchArrow based text
pre-processing. The main motivation for this example is to demonstrate the authoring of a text processing pipeline on
top of TorchArrow DataFrame.

## Installation and Usage

The example depends on TorchArrow and TorchData.

#### TorchArrow Installation

Install it from source following instructions at https://github.com/pytorch/torcharrow#from-source. Note that some of
the natively integrated text operators (`bpe_tokenize` for tokenization, `lookup_indices` for vocabulary look-up) used
in this example depend on the torch library. By default, TorchArrow doesn’t take dependency on the torch library. Hence
make sure to use flag `USE_TORCH=1` during TorchArrow installation (this is also the reason why we cannot depend on
nightly releases)

```
USE_TORCH=1 python setup.py install
```

#### TorchData Installation

To install TorchData follow instructions athttps://github.com/pytorch/data#installation

#### Usage

To run example from command line run following command:

```bash
python roberta_sst2_training_with_torcharrow.py \
        --batch-size 16 \
        --num-epochs 1 \
        --learning-rate 1e-5
```
