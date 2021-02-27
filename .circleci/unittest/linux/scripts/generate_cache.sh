#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

pytest test/common/cache_utils.py