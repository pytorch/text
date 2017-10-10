def get_tokenizer(tokenizer):
    if callable(tokenizer):
        return tokenizer
    if tokenizer == "spacy":
        try:
            import spacy
            spacy_en = spacy.load('en')
            return lambda s: [tok.text for tok in spacy_en.tokenizer(s)]
        except ImportError:
            print("Please install SpaCy and the SpaCy English tokenizer. "
                  "See the docs at https://spacy.io for more information.")
            raise
        except AttributeError:
            print("Please install SpaCy and the SpaCy English tokenizer. "
                  "See the docs at https://spacy.io for more information.")
            raise
    elif tokenizer == "moses":
        try:
            from nltk.tokenize.moses import MosesTokenizer
            moses_tokenizer = MosesTokenizer()
            return moses_tokenizer.tokenize
        except ImportError:
            print("Please install NLTK. "
                  "See the docs at http://nltk.org for more information.")
            raise
        except LookupError:
            print("Please install the necessary NLTK corpora. "
                  "See the docs at http://nltk.org for more information.")
            raise
    elif tokenizer == 'revtok':
        try:
            import revtok
            return revtok.tokenize
        except ImportError:
            print("Please install revtok.")
            raise
    elif tokenizer == 'subword':
        try:
            import revtok
            return lambda x: revtok.tokenize(x, decap=True)
        except ImportError:
            print("Please install revtok.")
            raise
    raise ValueError("Requested tokenizer {}, valid choices are a "
                     "callable that takes a single string as input, "
                     "\"revtok\" for the revtok reversible tokenizer, "
                     "\"subword\" for the revtok caps-aware tokenizer, "
                     "\"spacy\" for the SpaCy English tokenizer, or "
                     "\"moses\" for the NLTK port of the Moses tokenization "
                     "script.".format(tokenizer))


def interleave_keys(a, b):
    """Interleave bits from two sort keys to form a joint sort key.

    Examples that are similar in both of the provided keys will have similar
    values for the key defined by this function. Useful for tasks with two
    text fields like machine translation or natural language inference.
    """
    def interleave(args):
        return ''.join([x for t in zip(*args) for x in t])
    return int(''.join(interleave(format(x, '016b') for x in (a, b))), base=2)
