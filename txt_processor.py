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


_p1 = [r'<.*>', '&amp;', '&lt;', '&gt;',
       r'<ref[^<]*<\/ref>', r'<[^>]*>',
       r'\[http:[^] ]*', r'\|thumb', r'\|left', r'\|right', r'\|\d+px',
       r'\[\[image:[^\[\]]*\|', r'\[\[category:([^|\]]*)[^]]*\]\]',
       r'\[\[[a-z\-]*:[^\]]*\]\]', r'\[\[[^\|\]]*\|', r'\{\{[^\}]*\}\}',
       r'\{[^\}]*\}', r'\[', r'\]', r'&[^;]*;',
       'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
       'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
       '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
       r'[^0-9a-zA-Z\n]+', r'\n ', r'\n\s*\n']
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

_patterns = [r'\'',
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

_replacements = [' \'  ',
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

_patterns_dict = list((re.compile(p), r) for p, r in zip(_p1, _r1))


def _basic_english_normalize(line):
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

    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line.split()


#getLines(20, 5)
with open("enwik9_short.txt") as f:
    for line in range(200):
        line = f.readline()
        print(_basic_english_normalize(line))
