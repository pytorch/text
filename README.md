# This is a temp repo for hack week: Data APIs for NLP

## Get started
* install HuggingFace datasets. We copied it here to jump start. Eventually, we will build our own.
> pip install -e stl_text/dataframes/datasets

* install PyTorch and torchtext nightlies as some of the tasks depend on the prototype work in torchtext library.

to install cpu version on Linux:
> pip install --pre torch torchtext -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html;

to install cuda 10.1 version on Linux:
> pip install --pre torch torchtext -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html;

More detailed instructions are available [here](https://pytorch.org/).

* install this package
> pip install -e .

* run an example
> python examples/hf_dataset_quick_tour.py
