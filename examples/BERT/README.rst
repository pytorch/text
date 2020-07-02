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
