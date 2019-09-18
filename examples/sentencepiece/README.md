# This is an example to create a dataset with SentencePiece binding. 

In this example, the training data in the raw file is used to train a
sentencepiece model with the unigram method. The pretrained tokenizer is
used to process both training and testing data for the dataset. A text
sentiment model is developed and applied to reproduce the YelpReviewFull
results from fastText. 

To try the example, simply run the bash script below:

```bash 
./run_script.sh
```
