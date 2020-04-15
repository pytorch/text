import torch
from torch._jit_internal import Tuple


Tensor = torch.Tensor


class MultiheadAttentionContainer(torch.nn.Module):
    def __init__(self, in_proj, attention_layer, out_proj):
        r"""Process input using multi-head attention.
        Args:
            attention_layer: The attention layer. The default is None and scaled dot product
                attention will be used.

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
        self.query_in_proj = in_proj[0]
        self.key_in_proj = in_proj[1]
        self.value_in_proj = in_proj[2]
        self.attention_layer = attention_layer
        self.out_proj = out_proj

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
        q = self.query_in_proj(query)
        k = self.key_in_proj(key)
        v = self.value_in_proj(value)
        attn_output, attn_output_weights = self.attention_layer(q, k, v, attn_mask=attn_mask,
                                                                key_padding_mask=key_padding_mask)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_output_weights


class MultiheadInProject(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadInProject, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.proj_layer = torch.nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=False)

    def forward(self, seq):
        seq_len, bsz, proj_dim = seq.size()
        seq = self.proj_layer(seq)
        seq = seq.reshape(seq_len, bsz * self.num_heads, self.head_dim)
        return seq


class MultiheadOutProject(torch.nn.Module):
    def __init__(self, head_dim, num_heads):
        super(MultiheadOutProject, self).__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.proj_layer = torch.nn.Linear(num_heads * head_dim, num_heads * head_dim, bias=False)

    def forward(self, seq):
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
            - query: :math:`(L, N * H, E / H)`
            - key: :math:`(S, N * H, E / H)`
            - value: :math:`(S, N * H, E / H)`
            - key_padding_mask: :math:`(N, S)`
            - attn_mask: :math:`(L, S)` or :math:`(N * H, L, S)`
            - Output: :math:`(L, N * H, E / H)`, :math:`(N * H, L, S)`
            where L is the target length, S is the source length, H is the number
            of attention heads, N is the batch size, and E is the embedding dimension.
        """
        tgt_len, batch_heads, head_dim = query.size()
        assert query.size(1) == key.size(1) == value.size(1), "Dimension 0 of query, key, value must be equal."
        assert batch_heads % self.num_heads == 0, "Dimension 0 of query, key, value must be divisible by num_heads"
        bsz = batch_heads // self.num_heads
        assert key.size() == value.size(), "Shape of key, value must match"
        assert query.size(-1) == key.size(-1), "The head dimension of query must be equal to that of key"

        src_len = key.size(0)

        # Scale query
        query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
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
        return attn_output.transpose(0, 1), attn_output_weights
