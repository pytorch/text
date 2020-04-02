import torch
from torch._overrides import has_torch_function, handle_torch_function
import torch.nn.functional as F
from torch._jit_internal import Optional, Tuple


Tensor = torch.Tensor


def multi_head_attention_in_projection(seq, num_heads, in_proj_weight, in_proj_bias=None):
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
    if not torch.jit.is_scripting():
        tens_ops = (seq, in_proj_weight)
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(
                multi_head_attention_in_projection, tens_ops,
                seq, num_heads, in_proj_weight, in_proj_bias=in_proj_bias)
    seq_len, bsz, _ = seq.size()
    proj_dim = in_proj_weight.size(0)
    assert proj_dim % num_heads == 0, "projection dimension must be divisible by num_heads"
    head_dim = proj_dim // num_heads

    q = F.linear(seq, in_proj_weight, in_proj_bias)
    # Shape of q: (S, N, P)
    q = q.reshape(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
    return q


def scaled_dot_product_attention(q,                         # type: Tensor
                                 k,                         # type: Tensor
                                 v,                         # type: Tensor
                                 num_heads,                 # type: int
                                 add_zero_attn,             # type: bool
                                 dropout_p,                 # type: float
                                 training=True,             # type: bool
                                 key_padding_mask=None,     # type: Optional[Tensor]
                                 attn_mask=None,            # type: Optional[Tensor]
                                 ):
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
        - q: :math:`(N * H, L, P / H)`
        - k: :math:`(N * H, S, P / H)`
        - v: :math:`(N * H, S, P / H)`
        - key_padding_mask: :math:`(N, S)`
        - attn_mask: :math:`(L, S)` or :math:`(N * H, L, S)`
        - Output: :math:`(N * H, L, P / H)`, :math:`(N * H, L, S)`
        where L is the target length, S is the source length, H is the number
        of attention heads, N is the batch size, and P is the projection
        dimension.
    """
    if not torch.jit.is_scripting():
        tens_ops = (q, k, v)
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(
                scaled_dot_product_attention, tens_ops,
                q, k, v, num_heads, add_zero_attn, dropout_p,
                training=training, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
    batch_heads, tgt_len, head_dim = q.size()
    assert q.size(0) == k.size(0) == v.size(0), "Dimension 0 of q, k, v must be equal."
    assert batch_heads % num_heads == 0, "Dimension 0 of q, k, v must be divisible by num_heads"
    bsz = batch_heads // num_heads
    assert k.size() == v.size(), "Shape of k, v must match"
    assert q.size(-1) == k.size(-1), "The head dimension of query must be equal to that of key"

    src_len = k.size(1)

    # Scale q
    q = q * (float(head_dim) ** -0.5)
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
                attn_mask, torch.tensor(float('-inf')), torch.tensor(0.)).to(dtype=q.dtype, device=q.device)

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # Dot product of q, k
    attn_output_weights = torch.matmul(q, k.transpose(-2, -1))
    assert list(attn_output_weights.size()) == [batch_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.reshape(batch_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)

    attn_output = torch.matmul(F.dropout(attn_output_weights, p=dropout_p, training=training), v)
    return attn_output, attn_output_weights


def multi_head_attention_out_projection(attn_output, num_heads, out_proj_weight, out_proj_bias=None):
    # type: (Tensor, int, Tensor, Optional[Tensor]) -> Tensor
    r"""Projects an output sequence using parallel attention heads.
    Args:
        attn_output (Tensor): Projection to be decoded to an embedding.
        num_heads (int): Number of parallel attention heads
        out_proj_weight (Tensor): weight used to decode projection.
        out_proj_bias (Tensor, optional): bias used to decode projection.
    Shape:
        - attn_output: :math:`(N * H, S, P / H)`
        - out_proj_weight: :math:`(E, P)`
        - out_proj_bias: :math:`(E)`
        - Output: :math:`(S, N, E)`
        where S is the sequence length, H is the number of attention heads, N is the
        batch size, P is the projection dimension, and E is the embedding
        dimension.
    """
    if not torch.jit.is_scripting():
        tens_ops = (attn_output, out_proj_weight)
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(
                multi_head_attention_out_projection, tens_ops,
                attn_output, num_heads, out_proj_weight, out_proj_bias=out_proj_bias)
    batch_heads, seq_len, head_dim = attn_output.size()
    # embed_dim = out_proj_weight.size(0)
    assert batch_heads % num_heads == 0, "dimension 0 of attn_output must be divisible by num_heads"
    bsz = batch_heads // num_heads
    attn_output = attn_output.transpose(0, 1).reshape(seq_len, bsz, head_dim * num_heads)
    return F.linear(attn_output, out_proj_weight, out_proj_bias)
