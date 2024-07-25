# This is an example to train a text classification model

In the basic case, users can train the sentiment model in model.py with AG_NEWS dataset in torchtext.datasets.

To try the example, run the following script:

```bash
./run_script.sh
```

In addition, one can also use sentencepiece tokenizer as shown below. A text classification model is developed and
applied to reproduce the YelpReviewFull results from fastText.

To try the example, simply run the following commands:

```bash
python train.py YelpReviewFull --device cuda --use-sp-tokenizer True --num-epochs 10 --embed-dim 64
```
