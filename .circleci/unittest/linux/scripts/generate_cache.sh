#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

cd test
python -m torchtext_unittest.common.cache_utils
