import time
time1 = time.time()
import re
from torchtext.data.functional import custom_replace

def generate_offsets(filename):

    time0 = time.time()
    offsets = []
    with open(filename) as f:
        while f.readline():
#        for line in range(100):
            offsets.append(f.tell())
#            f.readline()
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


buffer_lines = []
input_filename = "enwik9_8000.txt"
output_filename = "NORMAL_enwik9_8000.txt"

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
print("total time: ", time.time() - time1)
time2 = time.time()
getLines(output_filename, generate_offsets(output_filename), 200, 10)
print("total time: ", time.time() - time2)
