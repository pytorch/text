#!/usr/bin/env bash

set -e

eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"
conda activate ./env

pytest test/common/cache_utils.py