import os
import logging

import tempfile

from torchtext.datasets.text_classification import URLS
from torchtext.utils import extract_archive
from torchtext.utils import download_from_url

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_path = '/tmp/asdf'
    with tempfile.NamedTemporaryFile() as tmp_file:
        extract_archive(download_from_url(URLS['AmazonReviewFull'], tmp_file.name), data_path)
