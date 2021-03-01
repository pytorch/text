#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="$(git rev-parse --show-toplevel)"

cd "${root_dir}"

eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"
conda activate ./env

printf "* Installing PyTorch\n"
conda install -y -c "pytorch-${UPLOAD_CHANNEL}" ${CONDA_CHANNEL_FLAGS} pytorch torchtext cpuonly

printf "* Installing parameterized\n"
pip install parameterized
