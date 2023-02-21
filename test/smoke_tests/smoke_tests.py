"""Run smoke tests"""

import os
import re

import torchdata
import torchtext
import torchtext.version  # noqa: F401

NIGHTLY_ALLOWED_DELTA = 3
channel = os.getenv("MATRIX_CHANNEL")


def validateTorchdataVersion():
    from datetime import datetime

    date_t_str = re.findall(r"dev\d+", torchdata.__version__)[0]
    date_t_delta = datetime.now() - datetime.strptime(date_t_str[3:], "%Y%m%d")

    if date_t_delta.days >= NIGHTLY_ALLOWED_DELTA:
        raise RuntimeError(f"torchdata binary {torchdata.__version__} is more than {NIGHTLY_ALLOWED_DELTA} days old!")


if channel == "nightly":
    validateTorchdataVersion()

print("torchtext version is ", torchtext.__version__)
print("torchdata version is ", torchdata.__version__)
