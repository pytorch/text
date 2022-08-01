"""
CNNDM Text Summarization with T5-Base model
=======================================================

**Author**: `Pendo Abbo <pabbo@fb.com>`__

"""

######################################################################
# Overview
# --------
#
# This tutorial demonstrates how to use a pre-trained T5 Model for text summarization on the CNN-DailyMail dataset.
# We will demonstrate how to use the torchtext library to:

# 1. Build a text pre-processing pipeline for a T5 model
# 2. Read in the CNNDM dataset and pre-process the text
# 3. Instantiate a pre-trained T5 model with base configuration, and perform text summarization on input text
#
#

######################################################################
# Common imports
# --------------
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#######################################################################
# Data Transformation
# -------------------
#
# The T5 model does not work with raw text. Instead, it requires the text to be transformed into numerical form
# in order to perform training and inference. The following transformations are required for the T5 model:

# 1. Tokenize text
# 2. Convert tokens into (integer) IDs
# 3. Truncate the sequences to a specified maximum length
# 4. Add end-of-sequence (EOS) and padding token IDs

# T5 uses a SentencePiece model for text tokenization. Below, we use a pre-trained SentencePiece model to build
# the text pre-processing pipeline using torchtext's `T5Transform`. Note that the transform supports both
# batched and non-batched text input (i.e. one can either pass a single sentence or a list of sentences), however
# the T5 model expects the input to be batched.

from torchtext.prototype.models import T5Transform

padding_idx = 0
eos_idx = 1
max_seq_len = 512
t5_sp_model_path = r"https://download.pytorch.org/models/text/t5_tokenizer_base.model"

transform = T5Transform(
    sp_model_path=t5_sp_model_path,
    max_seq_len=max_seq_len,
    eos_idx=eos_idx,
    padding_idx=padding_idx,
)

#######################################################################
# Alternatively, we can also use the transform shipped with the pre-trained models that does all of the above out-of-the-box
#
# ::
#
#   from torchtext.prototype.models import T5_BASE_GENERATION
#   transform = T5_BASE_GENERATION.transform()
#

#######################################################################
# Dataset
# -------
# torchtext provides several standard NLP datasets. For a complete list, refer to the documentation at https://pytorch.org/text/stable/datasets.html.
# These datasets are built using composable torchdata datapipes and hence support standard flow-control and mapping/transformation
# using user defined functions and transforms. Below, we demonstrate how to pre-process the CNNDM dataset to include the prefix necessary
# for the model to identify the task it is performing.

# The CNNDM dataset has a train, validation, and test split. Below we demo on the test split.
#
# .. note::
#       Using datapipes is still currently subject to a few caveats. If you wish
#       to extend this example to include shuffling, multi-processing, or
#       distributed learning, please see :ref:`this note <datapipes_warnings>`
#       for further instructions.

from functools import partial

from torch.utils.data import DataLoader
from torchtext.datasets.cnndm import CNNDM

batch_size = 5
test_datapipe = CNNDM(split="test")
task = "summarize"


def apply_prefix(task, x):
    return f"{task}: " + x[0], x[1]


test_datapipe = test_datapipe.map(partial(apply_prefix, task))
test_datapipe = test_datapipe.batch(batch_size)
test_datapipe = test_datapipe.rows2columnar(["article", "abstract"])
test_dataloader = DataLoader(test_datapipe, batch_size=None)

#######################################################################
# Alternately we can also use batched API (i.e apply the prefix on the whole batch)
#
# ::
#
#   def batch_prefix(task, x):
#    return {
#        "article": [f'{task}: ' + y for y in x["article"]],
#        "abstract": x["abstract"]
#    }
#
# batch_size = 5
# test_datapipe = CNNDM(split="test")
# task = 'summarize'
#
# test_datapipe = test_datapipe.batch(batch_size).rows2columnar(["article", "abstract"])
# test_datapipe = test_datapipe.map(partial(batch_prefix, task))
# test_dataloader = DataLoader(test_datapipe, batch_size=None)
#

######################################################################
# Model Preparation
# -----------------
#
# torchtext provides SOTA pre-trained models that can be used directly for NLP tasks or fine-tuned on downstream tasks. Below
# we use the pre-trained T5 model with standard base architecture to perform text summarization. For additional details on
# available pre-trained models, please refer to documentation at https://pytorch.org/text/main/models.html
#
#
from torchtext.prototype.models import T5_BASE_GENERATION


t5_base = T5_BASE_GENERATION
transform = t5_base.transform()
model = t5_base.get_model()
model.to(DEVICE)


#######################################################################
# Sequence Generator
# ------------------
#
# We can define a sequence generator to produce an output sequence based on the input sequence provided. This calls on the
# model's encoder and decoder, and iteratively expands the decoded sequences until the end-of-sequence token is generated
# for all sequences in the batch. The `greedy_generator` method shown below uses a greedy search (i.e. expands the sequence
# based on the most probable next word).
#

from torch import Tensor
from torchtext.prototype.models import T5Model


def greedy_generator(
    encoder_tokens: Tensor,
    eos_idx: int,
    model: T5Model,
) -> Tensor:

    # pass tokens through encoder
    encoder_padding_mask = encoder_tokens.eq(model.padding_idx)
    encoder_embeddings = model.dropout1(model.token_embeddings(encoder_tokens))
    encoder_output = model.encoder(encoder_embeddings, tgt_key_padding_mask=encoder_padding_mask)[0]

    encoder_output = model.norm1(encoder_output)
    encoder_output = model.dropout2(encoder_output)

    # initialize decoder input sequence; T5 uses padding index as starter index to decoder sequence
    decoder_tokens = torch.ones((encoder_tokens.size(0), 1), dtype=torch.long) * model.padding_idx

    # mask to keep track of sequences for which the decoder has not produced an end-of-sequence token yet
    incomplete_sentences = torch.ones((encoder_tokens.size(0), 1), dtype=torch.long)

    # iteratively generate output sequence until all sequences in the batch have generated the end-of-sequence token
    for step in range(model.config.max_seq_len):

        # causal mask and padding mask for decoder sequence
        tgt_len = decoder_tokens.shape[1]
        decoder_mask = torch.triu(torch.ones((tgt_len, tgt_len), dtype=torch.float64), diagonal=1).bool()
        decoder_padding_mask = decoder_tokens.eq(model.padding_idx)

        # T5 implemention uses padding idx to start sequence. Want to ignore this when masking
        decoder_padding_mask[:, 0] = False

        # pass decoder sequence through decoder
        decoder_embeddings = model.dropout3(model.token_embeddings(decoder_tokens))
        decoder_output = model.decoder(
            decoder_embeddings,
            memory=encoder_output,
            tgt_mask=decoder_mask,
            tgt_key_padding_mask=decoder_padding_mask,
            memory_key_padding_mask=encoder_padding_mask,
        )[0]

        decoder_output = model.norm2(decoder_output)
        decoder_output = model.dropout4(decoder_output)
        decoder_output = decoder_output * (model.config.embedding_dim ** -0.5)
        decoder_output = model.lm_head(decoder_output)

        # greedy search for next token to add to sequence
        probs = F.log_softmax(decoder_output[:, -1], dim=-1)
        _, next_token = torch.topk(probs, 1)

        # ignore next tokens for sentences that are already complete
        next_token *= incomplete_sentences

        # update incomplete_sentences to remove those that were just ended
        incomplete_sentences = incomplete_sentences - (next_token == eos_idx).long()

        # update decoder sequences to include new tokens
        decoder_tokens = torch.cat((decoder_tokens, next_token), 1)

        # early stop if all sentences have been ended
        if (incomplete_sentences == 0).all():
            break

    return decoder_tokens


#######################################################################
# Generate Summaries
# ------------------
#
# Finally we put all of the components together to generate summaries on the first batch of articles in the CNNDM test set.
#

batch = next(iter(test_dataloader))
input_text = batch["article"]
model_input = transform(input_text)
target = batch["abstract"]

model_output = greedy_generator(model=model, encoder_tokens=model_input, eos_idx=eos_idx)
output_text = transform.decode(model_output.tolist())

for i in range(batch_size):

    print(f"Example {i+1}:\n")
    print(f"greedy prediction: {output_text[i]}\n")
    print(f"target: {target[i]}\n\n")


#######################################################################
# Output
# ------
#
# ::
#
#    Example 1:
#
#    prediction: the Palestinians officially become the 123rd member of the international
#    criminal court . the move gives the court jurisdiction over alleged crimes committed
#    in the occupied Palestinian territory . the ICC opened a preliminary examination into
#    the situation in the occupied territories .
#
#    target: Membership gives the ICC jurisdiction over alleged crimes committed in
#    Palestinian territories since last June . Israel and the United States opposed the
#    move, which could open the door to war crimes investigations against Israelis .
#
#
#    Example 2:
#
#    prediction: a stray pooch in Washington state has used up at least three of her own
#   after being hit by a car . the dog staggers to a nearby farm, dirt-covered and
#    emaciated, where she is found . she suffered a dislocated jaw, leg injuries and a
#    caved-in sinus cavity .
#
#    target: Theia, a bully breed mix, was apparently hit by a car, whacked with a hammer
#    and buried in a field . "She's a true miracle dog and she deserves a good life," says
#    Sara Mellado, who is looking for a home for Theia .
#
#
#    Example 3:
#
#    prediction: mohammad Javad Zarif is the foreign minister of the country . he has been
#    a key figure in securing a breakthrough in nuclear talks . he has been a hero in the
#    international community .
#
#    target: Mohammad Javad Zarif has spent more time with John Kerry than any other
#    foreign minister . He once participated in a takeover of the Iranian Consulate in San
#    Francisco . The Iranian foreign minister tweets in English .
#
#
#    Example 4:
#
#    prediction: five americans were monitored for three weeks after being exposed to
#    Ebola . one of the five had a heart-related issue on Saturday and has been discharged .
#    none of the patients developed the deadly virus .
#
#    target: 17 Americans were exposed to the Ebola virus while in Sierra Leone in March .
#    Another person was diagnosed with the disease and taken to hospital in Maryland .
#    National Institutes of Health says the patient is in fair condition after weeks of
#    treatment .
#
#
#    Example 5:
#
#    prediction: the student was identified during an investigation by campus police and
#    the office of student affairs . he admitted to placing the noose on the tree early
#    Wednesday morning .
#
#    target: Student is no longer on Duke University campus and will face disciplinary
#    review . School officials identified student during investigation and the person
#    admitted to hanging the noose, Duke says . The noose, made of rope, was discovered on
#    campus about 2 a.m.
