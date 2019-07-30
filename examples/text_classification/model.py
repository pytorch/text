import torch.nn as nn

r'''
The model is composed of the embeddingbag layer and the linear layer.
nn.EmbeddingBag computes the mean of 'bags' of embeddings. Since it
doesn't instantiate the intermediate embeddings, nn.EmbeddingBag can
enhance the performance and memory efficiency to process a sequence
of tensors. Additionally, the text entries here have different lengths.
nn.EmbeddingBag requires no padding here so this method is much faster
than the original one with TorchText Iterator and Batch.

'''

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
        r'''
        Arguments:
            text: a bag of text tensors
            offsets: a list of offsets

        '''
        return self.fc(self.embedding(text, offsets))
