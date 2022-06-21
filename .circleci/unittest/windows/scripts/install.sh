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
conda install -y -c "pytorch-${UPLOAD_CHANNEL}" ${CONDA_CHANNEL_FLAGS} pytorch cpuonly

printf "* Installing torchdata nightly\n"
pip install --pre torchdata --extra-index-url https://download.pytorch.org/whl/nightly/cpu

printf "* Installing pywin32_postinstall script\n"
curl --output pywin32_postinstall.py https://raw.githubusercontent.com/mhammond/pywin32/main/pywin32_postinstall.py
python pywin32_postinstall.py -install

printf "* Installing torchtext\n"
git submodule update --init --recursive
"$root_dir/packaging/vc_env_helper.bat" python setup.py develop

printf "* Installing parameterized\n"
pip install parameterized
