import time
time1 = time.time()
import re
from torchtext.data.functional import custom_replace

import logging
import torch
import io
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
import os

def generate_offsets(filename):

    time0 = time.time()
    offsets = []
    with open(filename) as f:
        offsets.append(f.tell())
        while f.readline():
            offsets.append(f.tell())
    print("total time: ", time.time() - time0)
    return offsets


def getLines(filename, offsets, begin_line, num_lines):
    print(len(offsets))
    with open(filename) as f:
        f.seek(offsets[begin_line])
        for i in range(num_lines):
            print(f.readline())


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
             (r'A', 'a'), (r'B', 'b'), (r'C', 'c'), (r'D', 'd'), (r'E', 'e'), (r'F', 'f'),
             (r'G', 'g'), (r'H', 'h'), (r'I', 'i'), (r'J', 'j'), (r'K', 'k'), (r'L', 'l'),
             (r'M', 'm'), (r'N', 'n'), (r'O', 'o'), (r'P', 'p'), (r'Q', 'q'), (r'R', 'r'),
             (r'S', 's'), (r'T', 't'), (r'U', 'u'), (r'V', 'v'), (r'W', 'w'), (r'X', 'x'),
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


def simple_split(iterator):
    for line in iterator:
        yield line.split()


def read_lines_from_iterator(data_path, offsets, begin_line, num_lines):
    with open(data_path) as f:
        f.seek(offsets[begin_line])
        for i in range(num_lines):
            yield f.readline()


class EnWik9Dataset(torch.utils.data.Dataset):
    """Defines an abstract text classification datasets.
       Currently, we only support the following datasets:
    """

    def __init__(self, data):
        """Initiate text-classification dataset.
        Arguments:
            data: a list of label/tokens tuple. tokens are a tensor after
        Examples:
            See the examples in examples/text_classification/
        """

        super(EnWik9Dataset, self).__init__()
        self._data = data

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x


def normalized_raw_enwik9(input_filename, output_filename):
    with open(input_filename, 'r') as f1:
        with open(output_filename, 'w') as f2:
            while True:
                line = f1.readline()
                if not line:
                    break
                line = list(enwik9_norm_transform([line]))[0]
                if line != ' ' and line != '':
                    if line[0] == ' ':
                        line = line[1:]
                    f2.writelines(line + '\n')
    return


def _setup_datasets(begin_line, num_lines, root='.data'):

    raw_filename = os.path.join(root, 'enwik9_raw', 'enwik9_8000.txt')
    normalized_filename = os.path.join(root, 'enwik9_raw', 'NORMAL_enwik9_8000.txt')

    if not os.path.exists(normalized_filename):
        normalized_raw_enwik9(raw_filename, normalized_filename)

    offsets = generate_offsets(normalized_filename)
    read_lines = read_lines_from_iterator(normalized_filename,
                                          offsets, begin_line, num_lines)

    _data = []
    for item in simple_split(read_lines):
        _data += item

    return EnWik9Dataset(_data)


print(_setup_datasets(0, 5)._data)
