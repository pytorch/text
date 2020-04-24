#!/usr/bin/env python3

"""
To compare new version with previous:

    ./regenerate.sh
    meld <(git show HEAD:./config.yml | ./sort-yaml.py) <(cat config.yml | ./sort-yaml.py)

"""


import sys
import yaml

sys.stdout.write(yaml.dump(yaml.load(sys.stdin, Loader=yaml.FullLoader), sort_keys=True))
