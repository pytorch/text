def get_tokenizer(tokenizer):
    if callable(tokenizer):
        return tokenizer
    if tokenizer == 'spacy':
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
    raise ValueError("Requested tokenizer {}, valid choices are a "
                     "callable that takes a single string as input "
                     "and \"spacy\" for the SpaCy English "
                     "tokenizer.".format(tokenizer))


def interleave_keys(a, b):
    """Interleave bits from two sort keys to form a joint sort key.

    Examples that are similar in both of the provided keys will have similar
    values for the key defined by this function. Useful for tasks with two
    text fields like machine translation or natural language inference.
    """
    def interleave(args):
        return ''.join([x for t in zip(*args) for x in t])
    return int(''.join(interleave(format(x, '016b') for x in (a, b))), base=2)
