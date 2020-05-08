from typing import List
import torch


class TokenizerTransform(torch.nn.Module):
    r""" Initiate a tokenizer transform class
    """

    def __init__(self, tokenizer, language='en'):
        super(TokenizerTransform, self).__init__()
        if tokenizer is None:
            def split_tokenizer(line: str) -> List[str]:  # noqa: F821
                re_tokens: List[str] = []
                for token in line.split():
                    if token != '':
                        re_tokens.append(token)
                return re_tokens
            self.tokenizer = split_tokenizer
        elif tokenizer == "basic_english":
            if language != 'en':
                raise ValueError("Basic normalization is only available for Enlish(en)")

            def basic_english_normalize(line: str) -> List[str]:
                line = line.lower().replace('\'', ' \' ').replace('\"', '').replace('.', ' . ')
                line = line.replace('<br>', ' ').replace(',', ' , ').replace('(', ' ( ')
                line = line.replace(')', ' ) ').replace('!', ' ! ').replace('?', ' ? ')
                line = line.replace(';', ' ').replace(':', ' ')
                line = " ".join(line.split())
                re_tokens: List[str] = []
                for token in line.split():
                    if token != '':
                        re_tokens.append(token)
                return re_tokens

            self.tokenizer = basic_english_normalize
        elif tokenizer == "spacy":
            try:
                import spacy
                spacy = spacy.load(language)

                def spacy_tokenize(x, spacy):
                    return [tok.text for tok in spacy.tokenizer(x)]

                self.tokenizer = partial(spacy_tokenize, spacy=spacy)
            except ImportError:
                print("Please install SpaCy. "
                      "See the docs at https://spacy.io for more information.")
                raise
            except AttributeError:
                print("Please install SpaCy and the SpaCy {} tokenizer. "
                      "See the docs at https://spacy.io for more "
                      "information.".format(language))
                raise
        elif callable(tokenizer):
            self.tokenizer = tokenizer
        else:
            raise ValueError("Requested tokenizer {} is not supported".format(tokenizer))

    def forward(self, txt: str) -> List[str]:
        return self.tokenizer(txt)
