# This is an example to create a text classification dataset and train a sentiment model. 

In the basic case, users can train the sentiment model in model.py with 
AG_NEWS dataset in torchtext.datasets.text_classification. The dataset is
default with the ngrams number of 2.

To try the example, run the following script:

```bash
./run_script.sh
```

In addition, the training data in the raw file can be used to train a
sentencepiece model with the subword method. The pretrained tokenizer is
used to process both training and testing data for the dataset. A text
classification model is developed and applied to reproduce the YelpReviewFull
results from fastText. 

To try the example, simply run the following commands:

```bash 
python train.py YelpReviewFull --device cuda --use-sp-tokenizer True --num-epochs 10 --embed-dim 64
```
