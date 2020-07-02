from typing import NamedTuple

import torch.nn as nn


class WordCharCNNEmbedding(nn.Module):
    """The character embedding is built upon CNN and pooling layer
    with dropout applied before the convolution and after the pooling.
    """

    def __init__(
        self,
        ntokens: int,
        char_embedding_dim: int = 30,
        char_padding_idx: int = 1,
        dropout: float = 0.5,
        kernel_size: int = 3,
        out_channels: int = 30,
        target_emb: int = 300,
        use_highway: bool = False,
    ):
        super(WordCharCNNEmbedding, self).__init__()
        self._use_highway = use_highway

        if self._use_highway and out_channels != target_emb:
            raise ValueError("out_channels and target_emb must be " "equal in highway setting")

        self.char_embedding = nn.Embedding(ntokens, char_embedding_dim, char_padding_idx)
        self.conv_embedding = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv1d(
                in_channels=char_embedding_dim,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size - 1,
            ),
            nn.AdaptiveMaxPool1d(1),
        )
        self.proj_layer = nn.Linear(out_channels, target_emb)
        self.out_dropout = nn.Dropout(p=dropout)
        self._char_padding_idx = char_padding_idx

        self.init_weights()

    def init_weights(self):
        """Initialize the weight of character embedding with xavier
        and reinitalize the padding vectors to zero
        """

        self.char_embedding.weight.data.uniform_(-0.1, 0.1)
        # Reinitialize vectors at padding_idx to have 0 value
        self.char_embedding.weight.data[self._char_padding_idx].uniform_(0, 0)

    def forward(self, chars):
        """Run the forward calculation of the char-cnn embedding
        model.
        Args:
            chars (torch.Tensor): An integer tensor with the size of
                [seq_len x batch x char_size]
        Returns:
            char_embedding_vec (torch.Tensor): An embedding tensor with
                the size of [batch x seq_len x out_channels]
        """
        char_embedding_vec = self.char_embedding(chars)
        # Reshape the character embedding to the size of
        # [batch * seq_len, char_len, char_dim]
        char_embedding_vec = char_embedding_vec.view(
            -1, char_embedding_vec.size(2), char_embedding_vec.size(3)
        ).contiguous()
        # Transpose the embedding into [batch * seq_len, char_dim, char_len]
        char_embedding_vec = char_embedding_vec.transpose(1, 2).contiguous()
        # Apply char embedding with dropout and convolution
        # layers so the dim now will be [batch * seq_len, out_channel, new_len]
        char_embedding_vec = self.conv_embedding(char_embedding_vec)
        char_embedding_vec = char_embedding_vec.squeeze(-1)
        # Revert the size back to [seq_len, batch, out_channel]
        char_embedding_vec = char_embedding_vec.view(chars.size(0), chars.size(1), -1).contiguous()
        char_embedding_vec = self.out_dropout(char_embedding_vec)
        proj_char_embedding_vec = self.proj_layer(char_embedding_vec)
        # Apply highway connection between projection layer and
        # pooling layer
        if self._use_highway:
            proj_char_embedding_vec += char_embedding_vec

        return proj_char_embedding_vec
