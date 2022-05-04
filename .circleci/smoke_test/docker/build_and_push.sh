#!/usr/bin/env bash

set -euo pipefail

datestr="$(date "+%Y%m%d")"
image="pytorch/torchtext_smoke_base:smoke_test-${datestr}"
docker build -t "${image}" .
docker push "${image}"
