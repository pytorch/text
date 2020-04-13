import torch
from torch._jit_internal import Optional, Tuple


Tensor = torch.Tensor


class MultiheadAttentionContainer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, attention_layer=None, dropout=0.0):
        r"""Process input using multi-head attention.
        Args:
            embed_dim (int): Input embedding dimension
            num_heads (int): Number of parallel attention heads.
            attention_layer: The attention layer. The default is None and scaled dot product
                attention will be used.
            dropout: the dropout value (default=0.1).

        Examples::
            >>> MHA = torchtext.models.MultiheadAttentionContainer(10, 5)
            >>> query = torch.rand((21, 64, 10))
            >>> key = value = torch.rand((16, 64, 10))
            >>> attn_output, attn_weights = MHA(query, key, value)
            >>> print(attn_output.shape)
            >>> torch.Size([21, 64, 10])
        """
        super(MultiheadAttentionContainer, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads when head_dim=None"
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_in_proj = torch.nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=False)
        self.key_in_proj = torch.nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=False)
        self.value_in_proj = torch.nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=False)
        if attention_layer:
            self.attention_layer = attention_layer
        else:
            self.attention_layer = ScaledDotProduct(num_heads, dropout=dropout)
        self.out_proj = torch.nn.Linear(num_heads * self.head_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        r"""Uses a scaled dot product with the projected key-value pair to update
        the projected query.

        Args:
            query, key, value (Tensor): map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            key_padding_mask (Tensor, optional): if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            attn_mask (Tensor, optional): 2D or 3D mask that prevents attention to certain positions.
                This is an additive mask (i.e. the values will be added to the attention layer). A 2D mask
                will be broadcasted for all the batches while a 3D mask allows to specify a different mask
                for the entries of each batch.

        Shape:
            - Inputs:
            - query: :math:`(L, N, E)`
            - key: :math:`(S, N, E)`
            - value: :math:`(S, N, E)`
            - attn_mask: 3D mask :math:`(N*num_heads, L, S)`
            - key_padding_mask: :math:`(N, S)`

            - Outputs:
            - attn_output: :math:`(L, N, E)`
            - attn_output_weights: :math:`(N*num_heads, L, S)`

            where where L is the target length, S is the sequence length, H is the number of attention heads,
                N is the batch size, and E is the embedding dimension.
        """
        seq_len, bsz, proj_dim = query.size()
        tgt_len = key.size(0)
        q = self.query_in_proj(query).reshape(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = self.key_in_proj(key).reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = self.value_in_proj(value).reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        attn_output, attn_output_weights = self.attention_layer(q, k, v, attn_mask=attn_mask,
                                                                key_padding_mask=key_padding_mask)
        attn_output = self.out_proj(attn_output.transpose(0, 1).reshape(seq_len, bsz, self.head_dim * self.num_heads))
        return attn_output, attn_output_weights


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

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # type: (...) -> Tuple[Tensor, Tensor]
        r"""Uses a scaled dot product with the projected key-value pair to update
        the projected query.

        Args:
            query (Tensor): Projected query
            key (Tensor): Projected key
            value (Tensor): Projected value
            key_padding_mask (Tensor, optional): Specified padding elements in the
                key will be ignored by the attention. This is a binary mask. When
                the value is True, the corresponding value on the attention layer
                will be set to :math:`-\inf`.
            attn_mask (Tensor, optional): 2D or 3D mask that prevents attention to
                certain positions. This is an additive mask (i.e. the values will
                be added to the attention layer). A 2D mask will be broadcasted for
                all the batches while a 3D mask allows to specify a different mask
                for the entries of each batch.
        Shape:
            - query: :math:`(N * H, L, E / H)`
            - key: :math:`(N * H, S, E / H)`
            - value: :math:`(N * H, S, E / H)`
            - key_padding_mask: :math:`(N, S)`
            - attn_mask: :math:`(L, S)` or :math:`(N * H, L, S)`
            - Output: :math:`(N * H, L, E / H)`, :math:`(N * H, L, S)`
            where L is the target length, S is the source length, H is the number
            of attention heads, N is the batch size, and E is the embedding dimension.
        """
        batch_heads, tgt_len, head_dim = query.size()
        assert query.size(0) == key.size(0) == value.size(0), "Dimension 0 of query, key, value must be equal."
        assert batch_heads % self.num_heads == 0, "Dimension 0 of query, key, value must be divisible by num_heads"
        bsz = batch_heads // self.num_heads
        assert key.size() == value.size(), "Shape of key, value must match"
        assert query.size(-1) == key.size(-1), "The head dimension of query must be equal to that of key"

        src_len = key.size(1)

        # Scale query
        query = query * (float(head_dim) ** -0.5)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, tgt_len, src_len]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [batch_heads, tgt_len, src_len]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
            # attn_mask's dim is 3 now.
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.where(
                    attn_mask, torch.tensor(float('-inf')), torch.tensor(0.)).to(dtype=query.dtype, device=query.device)

        src_len = key.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        # Dot product of q, k
        attn_output_weights = torch.matmul(query, key.transpose(-2, -1))
        assert list(attn_output_weights.size()) == [batch_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.reshape(batch_heads, tgt_len, src_len)

        attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = torch.nn.functional.dropout(attn_output_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_output_weights, value)
        return attn_output, attn_output_weights
