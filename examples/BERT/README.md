# BERT with torchtext

This example shows how to train a BERT model with PyTorch and torchtext only. Then, we fine-tune the pre-trained BERT for the question-answer task.


## Generate pre-trained BERT

Train the BERT model with masked language modeling task and next-sentence task. Run the tasks on a local GPU or CPU:

    python mlm_task.py
    python ns_task.py

or run the tasks on a SLURM powered cluster with Distributed Data Parallel (DDP):

    srun --label --ntasks-per-node=1 --time=4000 --mem-per-cpu=5120 --gres=gpu:8 --cpus-per-task 80 --nodes=1 --pty python mlm_task.py --parallel DDP  --log-interval 600  --dataset BookCorpus

    srun --label --ntasks-per-node=1 --time=4000 --mem-per-cpu=5120 --gres=gpu:8 --cpus-per-task 80 --nodes=1 --pty python ns_task.py --parallel DDP --bert-model mlm_bert.pt --dataset BookCorpus

The result ppl of mlm_task is 18.97899 for the test set.
The result loss of ns_task is 0.05446 for the test set.

## Fine-tune pre-trained BERT for question-answer task

With SQuAD dataset, the pre-trained BERT is used for question-answer task:

    python qa_task.py  --bert-model ns_bert.pt --epochs 30

The pre-trained BERT models and vocab are available:

* [torchtext_bert_vocab.pt](https://pytorch.s3.amazonaws.com/models/text/torchtext_bert_example/torchtext_bert_vocab.pt)
* [mlm_bert.pt](https://pytorch.s3.amazonaws.com/models/text/torchtext_bert_example/mlm_bert.pt)
* [ns_bert.pt](https://pytorch.s3.amazonaws.com/models/text/torchtext_bert_example/ns_bert.pt)

An example train/valid/test printout with the pretrained BERT model in question-answer task:

    | epoch   1 |   200/ 1055 batches | lr 5.00000 | ms/batch 748.82 | loss  3.75 | ppl    42.32
    | epoch   1 |   400/ 1055 batches | lr 5.00000 | ms/batch 746.04 | loss  3.46 | ppl    31.85
    | epoch   1 |   600/ 1055 batches | lr 5.00000 | ms/batch 748.82 | loss  3.09 | ppl    21.90
    | epoch   1 |   800/ 1055 batches | lr 5.00000 | ms/batch 743.96 | loss  2.77 | ppl    15.89
    | epoch   1 |  1000/ 1055 batches | lr 5.00000 | ms/batch 743.21 | loss  2.30 | ppl     9.99
    -----------------------------------------------------------------------------------------
    | end of epoch   1 | time: 821.76s | valid loss  1.92 | exact   49.945% | f1   62.056%
    -----------------------------------------------------------------------------------------
    | epoch   2 |   200/ 1055 batches | lr 5.00000 | ms/batch 749.20 | loss  1.81 | ppl     6.10
    | epoch   2 |   400/ 1055 batches | lr 5.00000 | ms/batch 743.78 | loss  1.72 | ppl     5.61
    | epoch   2 |   600/ 1055 batches | lr 5.00000 | ms/batch 744.54 | loss  1.66 | ppl     5.28
    | epoch   2 |   800/ 1055 batches | lr 5.00000 | ms/batch 744.99 | loss  1.64 | ppl     5.17
    | epoch   2 |  1000/ 1055 batches | lr 5.00000 | ms/batch 744.06 | loss  1.60 | ppl     4.96
    -----------------------------------------------------------------------------------------
    | end of epoch   2 | time: 821.15s | valid loss  1.58 | exact   59.221% | f1   71.034%
    -----------------------------------------------------------------------------------------
    | epoch   3 |   200/ 1055 batches | lr 5.00000 | ms/batch 747.07 | loss  1.41 | ppl     4.10
    | epoch   3 |   400/ 1055 batches | lr 5.00000 | ms/batch 743.91 | loss  1.39 | ppl     4.03
    | epoch   3 |   600/ 1055 batches | lr 5.00000 | ms/batch 743.71 | loss  1.39 | ppl     4.03
    | epoch   3 |   800/ 1055 batches | lr 5.00000 | ms/batch 744.33 | loss  1.39 | ppl     4.03
    | epoch   3 |  1000/ 1055 batches | lr 5.00000 | ms/batch 744.86 | loss  1.40 | ppl     4.05
    -----------------------------------------------------------------------------------------
    | end of epoch   3 | time: 820.46s | valid loss  1.46 | exact   62.612% | f1   73.513%
    -----------------------------------------------------------------------------------------
    | epoch   4 |   200/ 1055 batches | lr 5.00000 | ms/batch 749.89 | loss  1.20 | ppl     3.33
    | epoch   4 |   400/ 1055 batches | lr 5.00000 | ms/batch 748.50 | loss  1.20 | ppl     3.32
    | epoch   4 |   600/ 1055 batches | lr 5.00000 | ms/batch 745.78 | loss  1.24 | ppl     3.47
    | epoch   4 |   800/ 1055 batches | lr 5.00000 | ms/batch 744.94 | loss  1.24 | ppl     3.45
    | epoch   4 |  1000/ 1055 batches | lr 5.00000 | ms/batch 744.22 | loss  1.25 | ppl     3.48
    -----------------------------------------------------------------------------------------
    | end of epoch   4 | time: 822.04s | valid loss  1.47 | exact   62.758% | f1   73.744%
    -----------------------------------------------------------------------------------------
    | epoch   5 |   200/ 1055 batches | lr 5.00000 | ms/batch 747.76 | loss  1.05 | ppl     2.87
    | epoch   5 |   400/ 1055 batches | lr 5.00000 | ms/batch 743.78 | loss  1.08 | ppl     2.94
    | epoch   5 |   600/ 1055 batches | lr 5.00000 | ms/batch 743.69 | loss  1.09 | ppl     2.97
    | epoch   5 |   800/ 1055 batches | lr 5.00000 | ms/batch 743.58 | loss  1.10 | ppl     3.01
    | epoch   5 |  1000/ 1055 batches | lr 5.00000 | ms/batch 743.05 | loss  1.13 | ppl     3.08
    -----------------------------------------------------------------------------------------
    | end of epoch   5 | time: 819.86s | valid loss  1.49 | exact   63.372% | f1   74.179%
    -----------------------------------------------------------------------------------------
    | epoch   6 |   200/ 1055 batches | lr 5.00000 | ms/batch 748.29 | loss  0.93 | ppl     2.54
    | epoch   6 |   400/ 1055 batches | lr 5.00000 | ms/batch 744.01 | loss  0.96 | ppl     2.62
    | epoch   6 |   600/ 1055 batches | lr 5.00000 | ms/batch 744.13 | loss  0.97 | ppl     2.63
    | epoch   6 |   800/ 1055 batches | lr 5.00000 | ms/batch 744.19 | loss  0.99 | ppl     2.68
    | epoch   6 |  1000/ 1055 batches | lr 5.00000 | ms/batch 744.10 | loss  1.00 | ppl     2.73
    -----------------------------------------------------------------------------------------
    | end of epoch   6 | time: 820.67s | valid loss  1.52 | exact   62.902% | f1   73.918%
    -----------------------------------------------------------------------------------------
    | epoch   7 |   200/ 1055 batches | lr 0.50000 | ms/batch 748.94 | loss  0.74 | ppl     2.09
    | epoch   7 |   400/ 1055 batches | lr 0.50000 | ms/batch 743.26 | loss  0.70 | ppl     2.01
    | epoch   7 |   600/ 1055 batches | lr 0.50000 | ms/batch 745.73 | loss  0.68 | ppl     1.97
    | epoch   7 |   800/ 1055 batches | lr 0.50000 | ms/batch 745.74 | loss  0.67 | ppl     1.96
    | epoch   7 |  1000/ 1055 batches | lr 0.50000 | ms/batch 744.42 | loss  0.65 | ppl     1.92
    -----------------------------------------------------------------------------------------
    | end of epoch   7 | time: 820.97s | valid loss  1.60 | exact   65.965% | f1   76.315%
    -----------------------------------------------------------------------------------------
    | epoch   8 |   200/ 1055 batches | lr 0.50000 | ms/batch 748.37 | loss  0.61 | ppl     1.85
    | epoch   8 |   400/ 1055 batches | lr 0.50000 | ms/batch 747.32 | loss  0.60 | ppl     1.82
    | epoch   8 |   600/ 1055 batches | lr 0.50000 | ms/batch 746.12 | loss  0.60 | ppl     1.82
    | epoch   8 |   800/ 1055 batches | lr 0.50000 | ms/batch 745.98 | loss  0.60 | ppl     1.83
    | epoch   8 |  1000/ 1055 batches | lr 0.50000 | ms/batch 744.58 | loss  0.60 | ppl     1.82
    -----------------------------------------------------------------------------------------
    | end of epoch   8 | time: 821.95s | valid loss  1.64 | exact   65.214% | f1   76.046%
    -----------------------------------------------------------------------------------------
    | epoch   9 |   200/ 1055 batches | lr 0.05000 | ms/batch 748.68 | loss  0.55 | ppl     1.74
    | epoch   9 |   400/ 1055 batches | lr 0.05000 | ms/batch 743.93 | loss  0.54 | ppl     1.71
    | epoch   9 |   600/ 1055 batches | lr 0.05000 | ms/batch 744.58 | loss  0.55 | ppl     1.72
    | epoch   9 |   800/ 1055 batches | lr 0.05000 | ms/batch 744.37 | loss  0.56 | ppl     1.75
    | epoch   9 |  1000/ 1055 batches | lr 0.05000 | ms/batch 744.40 | loss  0.54 | ppl     1.72
    -----------------------------------------------------------------------------------------
    | end of epoch   9 | time: 820.87s | valid loss  1.66 | exact   65.272% | f1   75.929%
    -----------------------------------------------------------------------------------------
    | epoch  10 |   200/ 1055 batches | lr 0.00500 | ms/batch 748.50 | loss  0.54 | ppl     1.72
    | epoch  10 |   400/ 1055 batches | lr 0.00500 | ms/batch 744.92 | loss  0.55 | ppl     1.72
    | epoch  10 |   600/ 1055 batches | lr 0.00500 | ms/batch 745.06 | loss  0.55 | ppl     1.73
    | epoch  10 |   800/ 1055 batches | lr 0.00500 | ms/batch 745.30 | loss  0.54 | ppl     1.71
    | epoch  10 |  1000/ 1055 batches | lr 0.00500 | ms/batch 746.06 | loss  0.54 | ppl     1.72
    -----------------------------------------------------------------------------------------
    | end of epoch  10 | time: 821.62s | valid loss  1.67 | exact   65.382% | f1   76.090%
    -----------------------------------------------------------------------------------------
    =========================================================================================
    | End of training | test loss  1.61 | exact   66.124% | f1   76.373% 
    =========================================================================================

## Structure of the example

### model.py

This file defines the Transformer and MultiheadAttention models used for BERT. The embedding layer include PositionalEncoding and TokenTypeEncoding layers. MLMTask, NextSentenceTask, and QuestionAnswerTask are the models for the three tasks mentioned above.

### data.py

This file provides a few datasets required to train the BERT model and question-answer task. Please note that BookCorpus dataset is not available publicly.


### mlm_task.py, ns_task.py, qa_task.py

Those three files define the train/valid/test process for the tasks.


### metrics.py

This file provides two metrics (F1 and exact score) for question-answer task


### utils.py

This file provides a few utils used by the three tasks.


## Multilingual Masked Language Modeling (MLM) with XLM-R

This example also shows a cross-lingual MLM task with XLM-R. The workflow requires a pretrained SentencePiece model for preprocessing the text strings, which is available to download from [link](https://pytorch.s3.amazonaws.com/models/text/torchtext_bert_example/sentencepiece.xlmr.model).

To run the workflow with 3000 lines from each of the 100 languages (CC-100 dataset)

    python cross_lingual_mlm_task.py --num_lines 3000

To run the distributed training use '--dist' flag, to specify world size use '--world_size=N', the default world size is 3 for one master and 2 worker nodes.

    python cross_lingual_mlm_task.py --num_lines 3000 --dist

To Run the reference XLM-R model from fairseq, download and unzip the pretrained model from [link](https://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.tar.gz).

    python cross_lingual_mlm_task.py --eval_ref ./xlmr.large 
