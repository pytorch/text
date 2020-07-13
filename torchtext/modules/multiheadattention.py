import torch
from typing import Tuple, Optional


class MultiheadAttentionContainer(torch.nn.Module):
    def __init__(self, nhead, in_proj_container, attention_layer, out_proj, batch_first=False):
        r""" A multi-head attention container

        Args:
            nhead: the number of heads in the multiheadattention model
            in_proj_container: A container of multi-head in-projection linear layers (a.k.a nn.Linear).
            attention_layer: The custom attention layer. The input sent from MHA container to the attention layer
                is in the shape of `(seq, batch, feature)` while the output shape of the attention layer
                is expected to be `(seq, batch, feature)`.
            out_proj: The multi-head out-projection layer (a.k.a nn.Linear).
            batch_first: If ``True``, then the input and output tensors are provided
                as `(batch, seq, feature)`. Default: ``False``

        Examples::
            >>> import torch
            >>> embed_dim, num_heads, bsz = 10, 5, 64
            >>> in_proj_container = InProjContainer(torch.nn.Linear(embed_dim, embed_dim),
                                                    torch.nn.Linear(embed_dim, embed_dim),
                                                    torch.nn.Linear(embed_dim, embed_dim))
            >>> MHA = MultiheadAttentionContainer(num_heads,
                                                  in_proj_container,
                                                  ScaledDotProduct(),
                                                  torch.nn.Linear(embed_dim, embed_dim))
            >>> query = torch.rand((21, bsz, embed_dim))
            >>> key = value = torch.rand((16, bsz, embed_dim))
            >>> attn_output, attn_weights = MHA(query, key, value)
            >>> print(attn_output.shape)
            >>> torch.Size([21, 64, 10])
        """
        super(MultiheadAttentionContainer, self).__init__()
        self.nhead = nhead
        self.in_proj_container = in_proj_container
        self.attention_layer = attention_layer
        self.out_proj = out_proj
        self.batch_first = batch_first

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                bias_k: Optional[torch.Tensor] = None,
                bias_v: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""

        Args:
            query, key, value (Tensor): map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            attn_mask, bias_k and bias_v (Tensor, optional): keyword arguments passed to the attention layer.
                See the definitions in the attention.

        Shape:
            - Inputs:
            - query: :math:`(L, N, E)`
            - key: :math:`(S, N, E)`
            - value: :math:`(S, N, E)`
            - attn_mask, bias_k and bias_v: same with the shape of the corresponding args in attention layer.

            - Outputs:
            - attn_output: :math:`(L, N, E)`
            - attn_output_weights: :math:`(N * H, L, S)`

            where where L is the target length, S is the sequence length, H is the number of attention heads,
                N is the batch size, and E is the embedding dimension.
        """
        if self.batch_first:
            query, key, value = query.transpose(-3, -2), key.transpose(-3, -2), value.transpose(-3, -2)

        tgt_len, src_len, bsz, embed_dim = query.size(-3), key.size(-3), query.size(-2), query.size(-1)
        q, k, v = self.in_proj_container(query, key, value)
        assert q.size(-1) % self.nhead == 0, "query's embed_dim must be divisible by the number of heads"
        head_dim = q.size(-1) // self.nhead
        q = q.reshape(tgt_len, bsz * self.nhead, head_dim)

        assert k.size(-1) % self.nhead == 0, "key's embed_dim must be divisible by the number of heads"
        head_dim = k.size(-1) // self.nhead
        k = k.reshape(src_len, bsz * self.nhead, head_dim)

        assert v.size(-1) % self.nhead == 0, "value's embed_dim must be divisible by the number of heads"
        head_dim = v.size(-1) // self.nhead
        v = v.reshape(src_len, bsz * self.nhead, head_dim)

        attn_output, attn_output_weights = self.attention_layer(q, k, v, attn_mask=attn_mask,
                                                                bias_k=bias_k, bias_v=bias_v)
        attn_output = attn_output.reshape(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(-3, -2)

        return attn_output, attn_output_weights


class ScaledDotProduct(torch.nn.Module):

    def __init__(self, dropout=0.0, batch_first=False):
        r"""Processes a projected query and key-value pair to apply
        scaled dot product attention.

        Args:
            dropout (float): probability of dropping an attention weight.
            batch_first: If ``True``, then the input and output tensors are provided
                as `(batch, seq, feature)`. Default: ``False``

        Examples::
            >>> SDP = torchtext.modules.ScaledDotProduct(dropout=0.1)
            >>> q = torch.randn(256, 21, 3)
            >>> k = v = torch.randn(256, 21, 3)
            >>> attn_output, attn_weights = SDP(q, k, v)
            >>> print(attn_output.shape, attn_weights.shape)
            torch.Size([256, 21, 3]) torch.Size([256, 21, 21])
        """
        super(ScaledDotProduct, self).__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                bias_k: Optional[torch.Tensor] = None,
                bias_v: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Uses a scaled dot product with the projected key-value pair to update
        the projected query.

        Args:
            query (Tensor): Projected query
            key (Tensor): Projected key
            value (Tensor): Projected value
            attn_mask (BoolTensor, optional): 3D mask that prevents attention to certain positions.
            bias_k and bias_v: (Tensor, optional): one more key and value sequence to be added at
                sequence dim (dim=-3). Those are used for incremental decoding. Users should provide
                non-None to both arguments in order to activate them.

        Shape:
            - query: :math:`(L, N * H, E / H)`
            - key: :math:`(S, N * H, E / H)`
            - value: :math:`(S, N * H, E / H)`
            - attn_mask: :math:`(N * H, L, S)`, positions with ``True`` are not allowed to attend
                while ``False`` values will be unchanged.
            - bias_k and bias_v:bias: :math:`(1, N * H, E / H)`

            - Output: :math:`(L, N * H, E / H)`, :math:`(N * H, L, S)`

            where L is the target length, S is the source length, H is the number
            of attention heads, N is the batch size, and E is the embedding dimension.
        """
        if self.batch_first:
            query, key, value = query.transpose(-3, -2), key.transpose(-3, -2), value.transpose(-3, -2)

        if bias_k is not None and bias_v is not None:
            assert key.size(-1) == bias_k.size(-1) and key.size(-2) == bias_k.size(-2) and bias_k.size(-3) == 1, \
                "Shape of bias_k is not supported"
            assert value.size(-1) == bias_v.size(-1) and value.size(-2) == bias_v.size(-2) and bias_v.size(-3) == 1, \
                "Shape of bias_v is not supported"
            key = torch.cat([key, bias_k])
            value = torch.cat([value, bias_v])
            if attn_mask is not None:
                attn_mask = torch.nn.functional.pad(attn_mask, (0, 1))

        tgt_len, head_dim = query.size(-3), query.size(-1)
        assert query.size(-1) == key.size(-1) == value.size(-1), "The feature dim of query, key, value must be equal."
        assert key.size() == value.size(), "Shape of key, value must match"
        src_len = key.size(-3)
        batch_heads = max(query.size(-2), key.size(-2))

        # Scale query
        query, key, value = query.transpose(-2, -3), key.transpose(-2, -3), value.transpose(-2, -3)
        query = query * (float(head_dim) ** -0.5)
        if attn_mask is not None:
            if attn_mask.dim() != 3:
                raise RuntimeError('attn_mask must be a 3D tensor.')
            if (attn_mask.size(-1) != src_len) or (attn_mask.size(-2) != tgt_len) or \
               (attn_mask.size(-3) != 1 and attn_mask.size(-3) != batch_heads):
                raise RuntimeError('The size of the attn_mask is not correct.')
            if attn_mask.dtype != torch.bool:
                raise RuntimeError('Only bool tensor is supported for attn_mask')

        # Dot product of q, k
        attn_output_weights = torch.matmul(query, key.transpose(-2, -1))
        if attn_mask is not None:
            attn_output_weights.masked_fill_(attn_mask, -1e8,)
        attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = torch.nn.functional.dropout(attn_output_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_output_weights, value)

        if self.batch_first:
            return attn_output, attn_output_weights
        else:
            return attn_output.transpose(-3, -2), attn_output_weights


class InProjContainer(torch.nn.Module):
    def __init__(self, query_proj, key_proj, value_proj):
        r"""A in-proj container to process inputs.

        Args:
            query_proj: a proj layer for query.
            key_proj: a proj layer for key.
            value_proj: a proj layer for value.
        """

        super(InProjContainer, self).__init__()
        self.query_proj = query_proj
        self.key_proj = key_proj
        self.value_proj = value_proj

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Projects the input sequences using in-proj layers.

        Args:
            query, key, value (Tensors): sequence to be projected

        Shape:
            - query, key, value: :math:`(S, N, E)`
            - Output: :math:`(S, N, E)`
            where S is the sequence length, N is the batch size, and E is the embedding dimension.
        """
        return self.query_proj(query), self.key_proj(key), self.value_proj(value)


def generate_square_subsequent_mask(nbatch, sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with True.
        Unmasked positions are filled with False.

    Args:
        nbatch: the number of batch size
        sz: the size of square mask
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).repeat(nbatch, 1, 1)
    return mask
