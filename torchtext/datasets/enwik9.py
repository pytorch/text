import logging
import os
from torchtext.data.functional import custom_replace
from torchtext.utils import (
    download_from_url,
    extract_archive,
    validate_file,
)
from torchtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
    _read_text_iterator,
)

URL = 'http://mattmahoney.net/dc/enwik9.zip'

MD5 = '3e773f8a1577fda2e27f871ca17f31fd'
MD5_processed = 'd854ec40bda3161c885c376654e15888'

NUM_LINES = {
    'train': 6348957
}

DATASET_NAME = "EnWik9"

_patterns = [(r'<.*>', ''),
             (r'&amp;', '&'),
             (r'&lt;', '<'),
             (r'&gt;', '>'),
             (r'<ref[^<]*<\/ref>', ''),
             (r'<[^>]*>', ''),
             (r'\[http:[^] ]*', '['),
             (r'\|thumb', ''),
             (r'\|left', ''),
             (r'\|right', ''),
             (r'\|\d+px', ''),
             (r'\[\[image:[^\[\]]*\|', ''),
             (r'\[\[category:([^|\]]*)[^]]*\]\]', '[[$1]]'),
             (r'\[\[[a-z\-]*:[^\]]*\]\]', ''),
             (r'\[\[[^\|\]]*\|', '[['),
             (r'\{\{[^\}]*\}\}', ''),
             (r'\{[^\}]*\}', ''),
             (r'\[', ''),
             (r'\]', ''),
             (r'&[^;]*;', ' '),
             (r'A', 'a'), (r'B', 'b'), (r'C', 'c'),
             (r'D', 'd'), (r'E', 'e'), (r'F', 'f'),
             (r'G', 'g'), (r'H', 'h'), (r'I', 'i'),
             (r'J', 'j'), (r'K', 'k'), (r'L', 'l'),
             (r'M', 'm'), (r'N', 'n'), (r'O', 'o'),
             (r'P', 'p'), (r'Q', 'q'), (r'R', 'r'),
             (r'S', 's'), (r'T', 't'), (r'U', 'u'),
             (r'V', 'v'), (r'W', 'w'), (r'X', 'x'),
             (r'Y', 'y'), (r'Z', 'z'),
             (r'0', ' zero '), (r'1', ' one '), (r'2', ' two '),
             (r'3', ' three '), (r'4', ' four '), (r'5', ' five '),
             (r'6', ' six '), (r'7', ' seven '), (r'8', ' eight '),
             (r'9', ' nine '),
             (r'[^a-z\n]+', ' '),
             (r'\n ', ''),
             (r'\s+', ' '),
             (r'\n\s*\n', r'\n')
             ]
enwik9_norm_transform = custom_replace(_patterns)


def preprocess_raw_enwik9(input_filename, output_filename):
    with open(input_filename, 'r') as f1:
        with open(output_filename, 'w') as f2:
            while True:
                line = f1.readline()
                if not line:
                    break
                if '#redirect' in line or '#REDIRECT' in line:
                    continue
                line = list(enwik9_norm_transform([line]))[0]
                if line != ' ' and line != '':
                    if line[0] == ' ':
                        line = line[1:]
                    f2.writelines(line + '\n')


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train',))
def EnWik9(root, split):
    logging.info('Creating {} data'.format(split))
    dataset_tar = download_from_url(URL, root=root, hash_value=MD5, hash_type='md5')
    extracted_files = extract_archive(dataset_tar)
    processed_file = os.path.join(root, 'norm_enwik9')
    if not(os.path.exists(processed_file) and validate_file(open(processed_file, 'rb'), MD5_processed, 'md5')):
        path = extracted_files[0]
        preprocess_raw_enwik9(path, processed_file)
    return _RawTextIterableDataset(DATASET_NAME,
                                   NUM_LINES[split], _read_text_iterator(processed_file))
