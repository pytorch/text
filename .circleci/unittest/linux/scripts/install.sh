#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

case "$(uname -s)" in
    Darwin*) os=MacOSX;;
    *) os=Linux
esac

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

printf "* Installing PyTorch\n"
(
    if [ "${os}" == MacOSX ] ; then
      # TODO: this can be removed as soon as linking issue could be resolved
      #  see https://github.com/pytorch/pytorch/issues/62424 from details
      MKL_CONSTRAINT='mkl==2021.2.0'
    else
      MKL_CONSTRAINT=''
    fi
    set -x
    conda install -y -c "pytorch-${UPLOAD_CHANNEL}" ${CONDA_CHANNEL_FLAGS} $MKL_CONSTRAINT pytorch cpuonly
)


printf "Installing torchdata nightly\n"
pip install --pre torchdata --extra-index-url https://download.pytorch.org/whl/nightly/cpu

printf "* Installing torchtext\n"
git submodule update --init --recursive
python setup.py develop

printf "* Installing parameterized\n"
pip install parameterized
