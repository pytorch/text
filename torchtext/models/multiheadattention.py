import torch
import torchtext.model.functional as F
from torch.nn.init import kaiming_uniform_
from math import sqrt


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
        >>> MHA_in = nn.MultiheadAttentionInProjection(10, 4, 3)
        >>> seq = torch.randn(21, 64, 10)
        >>> s = MHA_in(seq)
        >>> print(s.shape)
        torch.Size([256, 21, 3])
    """
    __constants__ = ['embed_dim', 'num_heads', 'head_dim']

    def __init__(self, embed_dim, num_heads, head_dim=None):
        super(MultiheadAttentionInProjection, self).__init__()
        if head_dim is None:
            assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads when head_dim=None"
            head_dim = embed_dim // num_heads
        self.head_dim = head_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.weight = torch.nn.Parameter(torch.Tensor(head_dim * num_heads, embed_dim))
        kaiming_uniform_(self.weight, a=sqrt(5))

    def forward(self, seq):
        return F.multi_head_attention_in_projection(seq, self.num_heads, self.weight, in_proj_bias=None)


class ScaledDotProduct(torch.nn.Module):
    r"""Processes a projected query and key-value pair to apply attention
    in each parallel attention head.
    Args:
        num_heads (int): Number of parallel attention heads.
        add_zero_attn (bool): Whether to add a batch of zeros to the key and
            value sequences.
        dropout_p (float): probability of dropping an attention weight.
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
        >>> SDP = nn.ScaledDotProduct(4, False, 0.1)
        >>> q = torch.randn(256, 21, 3)
        >>> k = v = torch.randn(256, 21, 3)
        >>> attn_output, attn_weights = SDP(q, k, v)
        >>> print(attn_output.shape, attn_weights.shape)
        torch.Size([256, 21, 3]) torch.Size([256, 21, 21])
    """
    __constants__ = ['num_heads', 'add_zero_attn', 'dropout_p']

    def __init__(self, num_heads, add_zero_attn=False, dropout_p=0.0):
        super(ScaledDotProduct, self).__init__()
        self.dropout_p = dropout_p
        self.add_zero_attn = add_zero_attn
        self.num_heads = num_heads

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        attn_output, attn_output_weights = F.scaled_dot_product_attention(
            query, key, value,
            self.num_heads, self.add_zero_attn, self.dropout_p, self.training, key_padding_mask, attn_mask)
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
            head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.weight = torch.nn.Parameter(torch.Tensor(embed_dim, head_dim * num_heads))
        kaiming_uniform_(self.weight, a=sqrt(5))

    def forward(self, attn_output):
        return F.multi_head_attention_out_projection(attn_output, self.num_heads, self.weight, out_proj_bias=None)
