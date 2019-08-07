from .utils import load_state_dict_from_url

import torch.nn as nn

__all__ = ['agnews', 'SogouNews', 'DBpedia', 'YelpReviewPolarity',
           'AmazonReviewPolarity']


model_urls = {'agnews': 'https://download.pytorch.org/' +
              'models/text/agnews-be5b303e.pt',
              'SogouNews': 'https://download.pytorch.org/' +
              'models/text/SogouNews-dd258c3e.pt',
              'DBpedia': 'https://download.pytorch.org/' +
              'models/text/DBpedia-86c06d71.pt',
              'YelpReviewPolarity': 'https://download.pytorch.org/' +
              'models/text/YelpReviewPolarity-62fd280d.pt',
              'AmazonReviewPolarity': 'https://download.pytorch.org/' +
              'models/text/AmazonReviewPolarity-1ec412da.pt'}


class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


def agnews(pretrained=False, progress=True, **kwargs):
    model = TextSentiment(1308844, 32, 4)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['agnews'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        return model


def SogouNews(pretrained=False, progress=True, **kwargs):
    model = TextSentiment(3121464, 32, 5)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['SogouNews'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        return model


def DBpedia(pretrained=False, progress=True, **kwargs):
    model = TextSentiment(6375026, 32, 14)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['DBpedia'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        return model


def YelpReviewPolarity(pretrained=False, progress=True, **kwargs):
    model = TextSentiment(6177283, 32, 2)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['YelpReviewPolarity'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        return model


def AmazonReviewPolarity(pretrained=False, progress=True, **kwargs):
    model = TextSentiment(20994453, 32, 2)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['AmazonReviewPolarity'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        return model
