from .utils import load_state_dict_from_url

import torch.nn as nn

__all__ = ['agnews']


model_urls = {'agnews': 'https://download.pytorch.org/' +
              'models/text/agnews-be5b303e.pt'}


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
