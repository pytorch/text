#!/usr/bin/env bash

set -e

eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"
conda activate ./env

python test/common/cache_utils.py