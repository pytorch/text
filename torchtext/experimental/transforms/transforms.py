from typing import List
import torch


class SplitTokenizer(torch.nn.Module):
    r""" Initiate a tokenizer transform class
    """

    def __init__(self):
        super(SplitTokenizer, self).__init__()

        def split_tokenizer(line: str) -> List[str]:  # noqa: F821
            re_tokens: List[str] = []
            for token in line.split():
                if token != '':
                    re_tokens.append(token)
            return re_tokens
        self.tokenizer = split_tokenizer

    def forward(self, txt: str) -> List[str]:
        return self.tokenizer(txt)


class BasicEnglishTokenizer(torch.nn.Module):
    r""" Initiate a tokenizer transform class
    """

    def __init__(self):
        super(BasicEnglishTokenizer, self).__init__()

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

    def forward(self, txt: str) -> List[str]:
        return self.tokenizer(txt)


class SpacyTokenizer(torch.nn.Module):
    r""" Initiate a tokenizer transform class
    """

    def __init__(self, language='en'):
        super(SpacyTokenizer, self).__init__()
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

    def forward(self, txt: str) -> List[str]:
        return self.tokenizer(txt)
