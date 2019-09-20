import time
time1 = time.time()
import re

byte_list = []
#with open("normalized_enwik9_short.txt") as f:
with open("enwik9_short.txt") as f:
    for line in range(100):
        byte_list.append(f.tell())
        f.readline()

def getLines(begin_line, num_lines):
    #with open("normalized_enwik9_short.txt") as f:
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
             (r'0', 'zero'), (r'1', 'one'), (r'2', 'two'),
             (r'3', 'three'), (r'4', 'four'), (r'5', 'five'),
             (r'6', 'six'), (r'7', 'seven'), (r'8', 'eight'),
             (r'9', 'nine'),
             (r'[^0-9a-zA-Z\n]+', ' '),
             (r'\n ', ''),
             (r'\n\s*\n', '\n'),
             (r'\s+', ' ')]

_r1 = ['', '&', '<', '>',
       '', '',
       '[', '', '', '', '',
       '', '[[$1]]',
       '', '[[', '',
       '', '', '', ' ',
       'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
       'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
       ' zero ', ' one ', ' two ', ' three ', ' four ', ' five ',
       ' six ', ' seven ', ' eight ', ' nine ',
       ' ', '', '\n']

_pattern = [r'\'',
            r'\"',
            r'\.',
            r'<br \/>',
            r',',
            r'\(',
            r'\)',
            r'\!',
            r'\?',
            r'\;',
            r'\:',
            r'\s+']

_replacement = [' \'  ',
                '',
                ' . ',
                ' ',
                ' , ',
                ' ( ',
                ' ) ',
                ' ! ',
                ' ? ',
                ' ',
                ' ',
                ' ']

_patterns_dict = list((re.compile(p), r) for (p, r) in _patterns)


def enwik9_normalize(line):
    r"""
    Basic normalization for a line of text.
    Normalization includes
    - lowercasing
    - complete some basic text normalization for English words as follows:
        add spaces before and after '\''
        remove '\"',
        add spaces before and after '.'
        replace '<br \/>'with single space
        add spaces before and after ','
        add spaces before and after '('
        add spaces before and after ')'
        add spaces before and after '!'
        add spaces before and after '?'
        replace ';' with single space
        replace ':' with single space
        replace multiple spaces with single space
    Returns a list of tokens after splitting on whitespace.
    """

#    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line


#getLines(20, 5)
buffer_lines = []
with open("enwik9", 'r') as f1:
    with open("NORMAL_enwik9.txt", 'w') as f2:
#        t1, t2, t3, t4, t5 = 0.0, 0.0, 0.0, 0.0, 0.0
        while True:
#        for _i in range(80000):
#            t1 = time.time()
            line = f1.readline()
#            t2 = time.time()
            if not line:
                break
#            t3 = time.time()
            line = enwik9_normalize(line)
#            t4 = time.time()
#            print(_i, line, len(line), line == ' ', line == '')
            if line != ' ' and line != '':
                if line[0] == ' ':
                    line = line[1:]

                buffer_lines.append(line + '\n')
#                f2.writelines(line + '\n')

            if len(buffer_lines) % 5000 == 0:
                f2.writelines(buffer_lines)
#            t5 = time.time()
#            print(t2 - t1, t3 - t2, t4 - t3, t5 - t4)
print("total time: ", time.time() - time1)
