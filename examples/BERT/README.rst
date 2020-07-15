BERT with torchtext
+++++++++

This example shows how to train a BERT model with PyTorch and torchtext only. Then, we fine-tune the pre-trained BERT for the question-answer task.


Generate pre-trained BERT
-------------------------

Train the BERT model with masked language modeling task and next-sentence task. Run the tasks on a local GPU or CPU:

    python mlm_task.py
    python ns_task.py

or run the tasks on a SLURM powered cluster with Distributed Data Parallel (DDP):

    srun --label --ntasks-per-node=1 --time=4000 --mem-per-cpu=5120 --gres=gpu:8 --cpus-per-task 80 --nodes=1 --pty python mlm_task.py --parallel DDP  --log-interval 600  --dataset BookCorpus

    srun --label --ntasks-per-node=1 --time=4000 --mem-per-cpu=5120 --gres=gpu:8 --cpus-per-task 80 --nodes=1 --pty python ns_task.py --parallel DDP --bert-model mlm_bert.pt --dataset BookCorpus

The result ppl of mlm_task is 18.97899 for the test set.
The result loss of ns_task is 0.05446 for the test set.

Fine-tune pre-trained BERT for question-answer task
---------------------------------------------------

With SQuAD dataset, the pre-trained BERT is used for question-answer task:

    python qa_task.py  --bert-model ns_bert.pt --epochs 30

The pre-trained BERT models and vocab are available:

* `bert_vocab.pt <https://pytorch.s3.amazonaws.com/models/text/torchtext_bert_example/bert_vocab.pt>`_
* `mlm_bert.pt <https://pytorch.s3.amazonaws.com/models/text/torchtext_bert_example/mlm_bert.pt>`_
* `ns_bert.pt <https://pytorch.s3.amazonaws.com/models/text/torchtext_bert_example/ns_bert.pt>`_

An example train/valid/test printout with the pretrained BERT model in question-answer task:

    | epoch   1 |   200/ 1055 batches | lr 5.00000 | ms/batch 746.33 | loss  3.70 | ppl    40.45
    | epoch   1 |   400/ 1055 batches | lr 5.00000 | ms/batch 746.78 | loss  3.06 | ppl    21.25
    | epoch   1 |   600/ 1055 batches | lr 5.00000 | ms/batch 746.83 | loss  2.84 | ppl    17.15
    | epoch   1 |   800/ 1055 batches | lr 5.00000 | ms/batch 746.55 | loss  2.69 | ppl    14.73
    | epoch   1 |  1000/ 1055 batches | lr 5.00000 | ms/batch 745.48 | loss  2.55 | ppl    12.85
    -----------------------------------------------------------------------------------------
    | end of epoch   1 | time: 821.25s | valid loss  2.33 | exact   40.052% | f1   52.595%
    -----------------------------------------------------------------------------------------
    | epoch   2 |   200/ 1055 batches | lr 5.00000 | ms/batch 748.17 | loss  2.33 | ppl    10.25
    | epoch   2 |   400/ 1055 batches | lr 5.00000 | ms/batch 745.52 | loss  2.28 | ppl     9.75
    | epoch   2 |   600/ 1055 batches | lr 5.00000 | ms/batch 745.50 | loss  2.24 | ppl     9.37
    | epoch   2 |   800/ 1055 batches | lr 5.00000 | ms/batch 745.10 | loss  2.22 | ppl     9.18
    | epoch   2 |  1000/ 1055 batches | lr 5.00000 | ms/batch 744.61 | loss  2.16 | ppl     8.66
    -----------------------------------------------------------------------------------------
    | end of epoch   2 | time: 820.75s | valid loss  2.12 | exact   44.632% | f1   57.965%
    -----------------------------------------------------------------------------------------
    | epoch   3 |   200/ 1055 batches | lr 5.00000 | ms/batch 748.88 | loss  2.00 | ppl     7.41
    | epoch   3 |   400/ 1055 batches | lr 5.00000 | ms/batch 746.46 | loss  1.99 | ppl     7.29
    | epoch   3 |   600/ 1055 batches | lr 5.00000 | ms/batch 745.35 | loss  1.99 | ppl     7.30
    | epoch   3 |   800/ 1055 batches | lr 5.00000 | ms/batch 746.03 | loss  1.98 | ppl     7.27
    | epoch   3 |  1000/ 1055 batches | lr 5.00000 | ms/batch 746.01 | loss  1.98 | ppl     7.26
    -----------------------------------------------------------------------------------------
    | end of epoch   3 | time: 821.98s | valid loss  1.96 | exact   48.001% | f1   61.036%
    -----------------------------------------------------------------------------------------
    | epoch   4 |   200/ 1055 batches | lr 5.00000 | ms/batch 748.72 | loss  1.82 | ppl     6.19
    | epoch   4 |   400/ 1055 batches | lr 5.00000 | ms/batch 745.86 | loss  1.84 | ppl     6.28
    | epoch   4 |   600/ 1055 batches | lr 5.00000 | ms/batch 745.63 | loss  1.85 | ppl     6.34
    | epoch   4 |   800/ 1055 batches | lr 5.00000 | ms/batch 745.59 | loss  1.82 | ppl     6.20
    | epoch   4 |  1000/ 1055 batches | lr 5.00000 | ms/batch 745.55 | loss  1.83 | ppl     6.21
    -----------------------------------------------------------------------------------------
    | end of epoch   4 | time: 821.10s | valid loss  1.95 | exact   49.149% | f1   62.040%
    -----------------------------------------------------------------------------------------
    | epoch   5 |   200/ 1055 batches | lr 5.00000 | ms/batch 748.40 | loss  1.66 | ppl     5.24
    | epoch   5 |   400/ 1055 batches | lr 5.00000 | ms/batch 756.09 | loss  1.69 | ppl     5.44
    | epoch   5 |   600/ 1055 batches | lr 5.00000 | ms/batch 769.19 | loss  1.70 | ppl     5.46
    | epoch   5 |   800/ 1055 batches | lr 5.00000 | ms/batch 764.96 | loss  1.72 | ppl     5.58
    | epoch   5 |  1000/ 1055 batches | lr 5.00000 | ms/batch 773.25 | loss  1.70 | ppl     5.49
    -----------------------------------------------------------------------------------------
    | end of epoch   5 | time: 844.20s | valid loss  1.99 | exact   49.509% | f1   61.994%
    -----------------------------------------------------------------------------------------
    | epoch   6 |   200/ 1055 batches | lr 0.50000 | ms/batch 765.25 | loss  1.50 | ppl     4.49
    | epoch   6 |   400/ 1055 batches | lr 0.50000 | ms/batch 749.64 | loss  1.45 | ppl     4.25
    | epoch   6 |   600/ 1055 batches | lr 0.50000 | ms/batch 768.16 | loss  1.40 | ppl     4.06
    | epoch   6 |   800/ 1055 batches | lr 0.50000 | ms/batch 745.69 | loss  1.43 | ppl     4.18
    | epoch   6 |  1000/ 1055 batches | lr 0.50000 | ms/batch 744.90 | loss  1.40 | ppl     4.07
    -----------------------------------------------------------------------------------------
    | end of epoch   6 | time: 829.55s | valid loss  1.97 | exact   51.182% | f1   63.437%
    -----------------------------------------------------------------------------------------
    | epoch   7 |   200/ 1055 batches | lr 0.50000 | ms/batch 747.73 | loss  1.36 | ppl     3.89
    | epoch   7 |   400/ 1055 batches | lr 0.50000 | ms/batch 744.50 | loss  1.37 | ppl     3.92
    | epoch   7 |   600/ 1055 batches | lr 0.50000 | ms/batch 744.20 | loss  1.35 | ppl     3.86
    | epoch   7 |   800/ 1055 batches | lr 0.50000 | ms/batch 743.85 | loss  1.36 | ppl     3.89
    | epoch   7 |  1000/ 1055 batches | lr 0.50000 | ms/batch 744.01 | loss  1.34 | ppl     3.83
    -----------------------------------------------------------------------------------------
    | end of epoch   7 | time: 820.02s | valid loss  2.01 | exact   51.507% | f1   63.885%
    -----------------------------------------------------------------------------------------
    | epoch   8 |   200/ 1055 batches | lr 0.50000 | ms/batch 747.40 | loss  1.31 | ppl     3.72
    | epoch   8 |   400/ 1055 batches | lr 0.50000 | ms/batch 744.33 | loss  1.30 | ppl     3.68
    | epoch   8 |   600/ 1055 batches | lr 0.50000 | ms/batch 745.76 | loss  1.31 | ppl     3.69
    | epoch   8 |   800/ 1055 batches | lr 0.50000 | ms/batch 745.04 | loss  1.31 | ppl     3.69
    | epoch   8 |  1000/ 1055 batches | lr 0.50000 | ms/batch 745.13 | loss  1.31 | ppl     3.72
    -----------------------------------------------------------------------------------------
    | end of epoch   8 | time: 820.40s | valid loss  2.02 | exact   51.260% | f1   63.762%
    -----------------------------------------------------------------------------------------
    | epoch   9 |   200/ 1055 batches | lr 0.05000 | ms/batch 748.36 | loss  1.26 | ppl     3.54
    | epoch   9 |   400/ 1055 batches | lr 0.05000 | ms/batch 744.55 | loss  1.26 | ppl     3.52
    | epoch   9 |   600/ 1055 batches | lr 0.05000 | ms/batch 745.46 | loss  1.23 | ppl     3.44
    | epoch   9 |   800/ 1055 batches | lr 0.05000 | ms/batch 745.23 | loss  1.26 | ppl     3.52
    | epoch   9 |  1000/ 1055 batches | lr 0.05000 | ms/batch 744.69 | loss  1.24 | ppl     3.47
    -----------------------------------------------------------------------------------------
    | end of epoch   9 | time: 820.41s | valid loss  2.02 | exact   51.578% | f1   63.704%
    -----------------------------------------------------------------------------------------
    | epoch  10 |   200/ 1055 batches | lr 0.00500 | ms/batch 749.25 | loss  1.25 | ppl     3.50
    | epoch  10 |   400/ 1055 batches | lr 0.00500 | ms/batch 745.81 | loss  1.24 | ppl     3.47
    | epoch  10 |   600/ 1055 batches | lr 0.00500 | ms/batch 744.89 | loss  1.26 | ppl     3.51
    | epoch  10 |   800/ 1055 batches | lr 0.00500 | ms/batch 746.02 | loss  1.23 | ppl     3.42
    | epoch  10 |  1000/ 1055 batches | lr 0.00500 | ms/batch 746.61 | loss  1.25 | ppl     3.50
    -----------------------------------------------------------------------------------------
    | end of epoch  10 | time: 821.85s | valid loss  2.05 | exact   51.648% | f1   63.811%
    -----------------------------------------------------------------------------------------
    =========================================================================================
    | End of training | test loss  2.05 | exact   51.337% | f1   63.645%
    =========================================================================================

Structure of the example
========================

model.py
--------

This file defines the Transformer and MultiheadAttention models used for BERT. The embedding layer include PositionalEncoding and TokenTypeEncoding layers. MLMTask, NextSentenceTask, and QuestionAnswerTask are the models for the three tasks mentioned above.

data.py
-------

This file provides a few datasets required to train the BERT model and question-answer task. Please note that BookCorpus dataset is not available publicly.


mlm_task.py, ns_task.py, qa_task.py
-----------------------------------

Those three files define the train/valid/test process for the tasks.


metrics.py
----------

This file provides two metrics (F1 and exact score) for question-answer task


utils.py
--------

This file provides a few utils used by the three tasks.
