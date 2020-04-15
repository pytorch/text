import torch
from torch._jit_internal import Tuple, Optional


Tensor = torch.Tensor


class MultiheadAttentionContainer(torch.nn.Module):
    def __init__(self, in_proj_tuple, attention_layer, out_proj):
        r""" A multi-head attention container

        Args:
            in_proj_tuple: A tuple of multi-head in-projection layers
            attention_layer: The attention layer.
            out_proj: The multi-head out-projection layer

        Examples::
            >>> embed_dim, num_heads, bsz = 10, 5, 64
            >>> MHA = MultiheadAttentionContainer((MultiheadInProject(embed_dim, num_heads),
                                                   MultiheadInProject(embed_dim, num_heads),
                                                   MultiheadInProject(embed_dim, num_heads)),
                                                   ScaledDotProduct(num_heads),
                                                   MultiheadOutProject(embed_dim // num_heads, num_heads))
            >>> query = torch.rand((21, bsz, embed_dim))
            >>> key = value = torch.rand((16, bsz, embed_dim))
            >>> attn_output, attn_weights = MHA(query, key, value)
            >>> print(attn_output.shape)
            >>> torch.Size([21, 64, 10])
        """
        super(MultiheadAttentionContainer, self).__init__()
        self.query_in_proj = in_proj_tuple[0]
        self.key_in_proj = in_proj_tuple[1]
        self.value_in_proj = in_proj_tuple[2]
        self.attention_layer = attention_layer
        self.out_proj = out_proj

    def forward(self, query, key, value, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""

        Args:
            query, key, value (Tensor): map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            attn_mask (Bool Tensor, optional): 3D mask that prevents attention to certain positions.

        Shape:
            - Inputs:
            - query: :math:`(L, N, E)`
            - key: :math:`(S, N, E)`
            - value: :math:`(S, N, E)`
            - attn_mask: :math:`(N * H, L, S)`

            - Outputs:
            - attn_output: :math:`(L, N, E)`
            - attn_output_weights: :math:`(N*num_heads, L, S)`

            where where L is the target length, S is the sequence length, H is the number of attention heads,
                N is the batch size, and E is the embedding dimension.
        """
        q = self.query_in_proj(query)
        k = self.key_in_proj(key)
        v = self.value_in_proj(value)
        attn_output, attn_output_weights = self.attention_layer(q, k, v, attn_mask=attn_mask)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_output_weights


class MultiheadInProject(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        r"""Process input using multi-head attention.

        Args:
            embed_dim (int): Input embedding dimension
            num_heads (int): Number of parallel attention heads.
        """

        super(MultiheadInProject, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.proj_layer = torch.nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=False)

    def forward(self, seq):
        # type: (Tensor) -> Tensor
        r"""Projects an input sequence using parallel attention heads.

        Args:
            seq (Tensor): sequence to be projected

        Shape:
            - seq: :math:`(S, N, E)`

            - Output: :math:`(S, N * H, E / H)`

            where S is the sequence length, H is the number of attention heads, N is the
            batch size, and E is the embedding dimension.
        """
        seq_len, bsz, proj_dim = seq.size()
        seq = self.proj_layer(seq)
        seq = seq.reshape(seq_len, bsz * self.num_heads, self.head_dim)
        return seq


class MultiheadOutProject(torch.nn.Module):
    def __init__(self, head_dim, num_heads):
        r"""Process attention output using multi-head attention.

        Args:
            head_dim (int): Dimension of embedding for each attention head.
            num_heads (int): Number of parallel attention heads.

        """
        super(MultiheadOutProject, self).__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.proj_layer = torch.nn.Linear(num_heads * head_dim, num_heads * head_dim, bias=False)

    def forward(self, seq):
        # type: (Tensor) -> Tensor
        r"""Projects an output sequence using parallel attention heads.

        Args:
            seq (Tensor): Projection to be decoded to an embedding.

        Shape:
            - seq: :math:`(S, N * H, E / H)`

            - Output: :math:`(S, N, E)`

            where S is the sequence length, H is the number of attention heads, N is the
            batch size, and E is the embedding dimension.
        """
        seq_len, bsz_num_head, head_dim = seq.size()
        assert bsz_num_head % self.num_heads == 0, \
            "Dimension -2 of MultiheadOutProject input must be divisible by num_heads"
        bsz = bsz_num_head // self.num_heads
        seq = seq.reshape(seq_len, bsz, self.num_heads * self.head_dim)
        seq = self.proj_layer(seq)
        return seq


class ScaledDotProduct(torch.nn.Module):
    __constants__ = ['num_heads', 'dropout']

    def __init__(self, num_heads, dropout=0.0):
        r"""Processes a projected query and key-value pair to apply
        scaled dot product attention.

        Args:
            num_heads (int): Number of parallel attention heads.
            dropout (float): probability of dropping an attention weight.

        Examples::
            >>> SDP = torchtext.models.ScaledDotProduct(4, 0.1)
            >>> q = torch.randn(256, 21, 3)
            >>> k = v = torch.randn(256, 21, 3)
            >>> attn_output, attn_weights = SDP(q, k, v)
            >>> print(attn_output.shape, attn_weights.shape)
            torch.Size([256, 21, 3]) torch.Size([256, 21, 21])
        """
        super(ScaledDotProduct, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, query, key, value, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""Uses a scaled dot product with the projected key-value pair to update
        the projected query.

        Args:
            query (Tensor): Projected query
            key (Tensor): Projected key
            value (Tensor): Projected value
            attn_mask (Bool Tensor, optional): 3D mask that prevents attention to certain positions.

        Shape:
            - query: :math:`(L, N * H, E / H)`
            - key: :math:`(S, N * H, E / H)`
            - value: :math:`(S, N * H, E / H)`
            - attn_mask: :math:`(N * H, L, S)`

            - Output: :math:`(L, N * H, E / H)`, :math:`(N * H, L, S)`

            where L is the target length, S is the source length, H is the number
            of attention heads, N is the batch size, and E is the embedding dimension.
        """
        tgt_len, batch_heads, head_dim = query.size()
        assert query.size(1) == key.size(1) == value.size(1), "Dimension 0 of query, key, value must be equal."
        assert batch_heads % self.num_heads == 0, "Dimension 0 of query, key, value must be divisible by num_heads"
        assert key.size() == value.size(), "Shape of key, value must match"
        assert query.size(-1) == key.size(-1), "The head dimension of query must be equal to that of key"
        src_len = key.size(0)

        # Scale query
        query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        query = query * (float(head_dim) ** -0.5)
        if attn_mask is not None:
            if list(attn_mask.size()) != [batch_heads, tgt_len, src_len]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
            if attn_mask.dtype != torch.bool:
                raise RuntimeError('Only bool tensor is supported for attn_mask')

        # Dot product of q, k
        attn_output_weights = torch.matmul(query, key.transpose(-2, -1))
        assert list(attn_output_weights.size()) == [batch_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'),)

        attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = torch.nn.functional.dropout(attn_output_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_output_weights, value)
        return attn_output.transpose(0, 1), attn_output_weights
