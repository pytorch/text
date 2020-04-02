import torch
from torch._jit_internal import Optional, Tuple


Tensor = torch.Tensor


class MultiheadAttentionInProjection(torch.nn.Module):
    r"""Process input using multi-head attention.
    Args:
        embed_dim (int): Input embedding dimension
        num_heads (int): Number of parallel attention heads.
        head_dim (int, optional): Dimension of embedding for each attention
            head. If not provided, then it is set to ``embed_dim / num_heads``.
    Shape:
        - seq: :math:`(S, N, E)`
        - Output: :math:`(N * H, S, D)`
        where S is the sequence length, N is the batch size, H is the number of
        attention heads, E is the embedding dimension, and D is the head
        dimension.
    Attributes:
        weight: The learnable weights of the module of shape
            :math:`(\text{head\_dim} * \text{num\_heads}, \text{embed\_dim})`.
    Examples::
        >>> # S = 21; N = 64; E = 10; D = 3; H = 4;
        >>> MHA_in = torchtext.models.MultiheadAttentionInProjection(10, 5)
        >>> seq = torch.randn(21, 64, 10)
        >>> s = MHA_in(seq)
        >>> print(s.shape)
        torch.Size([320, 21, 2])
    """
    __constants__ = ['embed_dim', 'num_heads', 'head_dim']

    def __init__(self, embed_dim, num_heads, head_dim=None):
        super(MultiheadAttentionInProjection, self).__init__()
        if head_dim is None:
            assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads when head_dim=None"
            self.head_dim = embed_dim // num_heads
        else:
            self.head_dim = head_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.linear = torch.nn.Linear(embed_dim, self.num_heads * self.head_dim)

    def forward(self, seq):
        # type: (Tensor, int, Tensor, Optional[Tensor]) -> Tensor
        r"""Projects an input sequence using parallel attention heads.
        Args:
            seq (Tensor): sequence to be projected
            num_heads (int): number of parallel heads used.
            in_proj_weight (Tensor): weight used for projection
            in_proj_bias (Tensor, optional): bias used for projection.
        Shape:
            - seq: :math:`(S, N, E)`
            - in_proj_weight: :math:`(P, E)`
            - in_proj_bias: :math:`(P)`
            - Output: :math:`(N * H, S, P / H)`
            where S is the sequence length, H is the number of attention heads, N is the
            batch size, P is the projection dimension, and E is the embedding
            dimension.
        """
        seq_len, bsz, proj_dim = seq.size()
        assert proj_dim % self.num_heads == 0, "projection dimension must be divisible by num_heads"
        head_dim = proj_dim // self.num_heads
        q = self.linear(seq)
        # Shape of q: (S, N, P)
        q = q.reshape(seq_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        return q


class ScaledDotProduct(torch.nn.Module):
    r"""Processes a projected query and key-value pair to apply attention
    in each parallel attention head.
    Args:
        num_heads (int): Number of parallel attention heads.
        dropout (float): probability of dropping an attention weight.
    Shape:
        - query: :math:`(N * H, L, D)`
        - key: :math:`(N * H, S, D)`
        - value: :math:`(N * H, S, D)`
        - key_padding_mask: :math:`(N, S)`
        - attn_mask: :math:`(L, S)` or :math:`(N * H, L, S)`
        - Output: :math:`(N * H, L, D)`, :math:`(N * H, L, S)`
        where L is the target sequence length, S is the source sequence
        length, H is the number of attention heads, N is the batch size,
        and D is the head dimension.
    Examples::
        >>> # S = L = 21; N = 64; E = 10; D = 3; H = 4;
        >>> SDP = torchtext.models.ScaledDotProduct(4, 0.1)
        >>> q = torch.randn(256, 21, 3)
        >>> k = v = torch.randn(256, 21, 3)
        >>> attn_output, attn_weights = SDP(q, k, v)
        >>> print(attn_output.shape, attn_weights.shape)
        torch.Size([256, 21, 3]) torch.Size([256, 21, 21])
    """
    __constants__ = ['num_heads', 'add_zero_attn', 'dropout_p']

    def __init__(self, num_heads, dropout=0.0):
        super(ScaledDotProduct, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        # type: (...) -> Tuple[Tensor, Tensor]
        r"""Uses a scaled dot product with the projected key-value pair to update
        the projected query.
        Args:
            q (Tensor): Projected query
            k (Tensor): Projected key
            v (Tensor): Projected value
            num_heads (int): Number of parallel attention heads.
            add_zero_attn (bool): Add a new batch of zeros to the projected key and
                value sequences at dimension 1.
            dropout_p (float): Probability of an element will be zeroed.
            training (bool): Apply dropout if ``training=True``
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
            - query: :math:`(N * H, L, P / H)`
            - key: :math:`(N * H, S, P / H)`
            - value: :math:`(N * H, S, P / H)`
            - key_padding_mask: :math:`(N, S)`
            - attn_mask: :math:`(L, S)` or :math:`(N * H, L, S)`
            - Output: :math:`(N * H, L, P / H)`, :math:`(N * H, L, S)`
            where L is the target length, S is the source length, H is the number
            of attention heads, N is the batch size, and P is the projection
            dimension.
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


class MultiheadAttentionOutProjection(torch.nn.Module):
    r"""Process attention output using multi-head attention.
    Args:
        embed_dim (int): Input projection dimension.
        num_heads (int): Number of parallel attention heads.
        head_dim (int, optional): Dimension of embedding for each attention
            head. If not provided, then it is set to ``embed_dim / num_heads``.
    Shape:
        - attn_output: :math:`(N * H, S, D)`
        - Output: :math:`(S, N, E)`
        where S is the sequence length, N is the batch size, H is the number of
        attention heads, E is the embedding dimension, and D is the head
        dimension.
    Attributes:
        weight: The learnable weights of the module of shape
            :math:`(\text{embed\_dim}, \text{head\_dim} * \text{num\_heads})`.
    Examples::
        >>> # S = 21; N = 64; E = 10; D = 3; H = 4;
        >>> MHA_out = nn.MultiheadAttentionOutProjection(10, 4, 3)
        >>> attn_seq = torch.randn(256, 21, 3)
        >>> a = MHA_out(attn_seq)
        >>> print(a.shape)
        torch.Size([21, 64, 10])
    """
    __constants__ = ['embed_dim', 'num_heads', 'head_dim']

    def __init__(self, embed_dim, num_heads, head_dim=None):
        super(MultiheadAttentionOutProjection, self).__init__()
        self.embed_dim = embed_dim
        if head_dim is None:
            assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads when head_dim=None"
            self.head_dim = embed_dim // num_heads
        else:
            self.head_dim = head_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.linear = torch.nn.Linear(self.num_heads * self.head_dim, embed_dim)

    def forward(self, attn_output):
        # type: (Tensor, int, Tensor, Optional[Tensor]) -> Tensor
        r"""Projects an output sequence using parallel attention heads.
        Args:
            attn_output (Tensor): Projection to be decoded to an embedding.
        Shape:
            - attn_output: :math:`(N * H, S, P / H)`
            where S is the sequence length, H is the number of attention heads, N is the
            batch size, P is the projection dimension, and E is the embedding
            dimension.
        """
        batch_heads, seq_len, head_dim = attn_output.size()
        # embed_dim = out_proj_weight.size(0)
        assert batch_heads % self.num_heads == 0, "dimension 0 of attn_output must be divisible by num_heads"
        bsz = batch_heads // self.num_heads
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, bsz, head_dim * self.num_heads)
        return self.linear(attn_output)
