#!/bin/bash
FILENAME=$1
URL=$2

wget -O - -o /dev/null $URL | sha256sum | head -c 64 > $FILENAME
