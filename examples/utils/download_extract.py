import logging
import argparse

from torchtext.utils import extract_archive
from torchtext.utils import download_from_url

parser = argparse.ArgumentParser(
    description='Download and extract a given dataset')
parser.add_argument('--url',
                    default='http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz')
parser.add_argument('--data', default='validation.tar.gz')
parser.add_argument('--logging-level', default='WARNING')
args = parser.parse_args()

logging.basicConfig(level=getattr(logging, args.logging_level))

tar_file = download_from_url(args.url, args.data)
extracted_files = extract_archive(args.data, 'extracted_files')
