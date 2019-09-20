import time
time1 = time.time()
import re

byte_list = []
with open("enwik9_short.txt") as f:
    for line in range(100):
        byte_list.append(f.tell())
        f.readline()


def getLines(begin_line, num_lines):
    with open("enwik9_short.txt") as f:
        f.seek(byte_list[begin_line])
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


_patterns_dict = list((re.compile(p), r) for (p, r) in _patterns)


def enwik9_normalize(line):
    r"""
    Basic normalization for a line of text.
    Returns a list of tokens after splitting on whitespace.
    """

    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line


buffer_lines = []
with open("enwik9_8000.txt", 'r') as f1:
    with open("NORMAL_enwik9_8000.txt", 'w') as f2:
        while True:
            line = f1.readline()
            if not line:
                break
            line = enwik9_normalize(line)
            if line != ' ' and line != '':
                if line[0] == ' ':
                    line = line[1:]
                f2.writelines(line + '\n')
print("total time: ", time.time() - time1)
