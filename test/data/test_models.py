import torch
from torchtext.models import MultiheadAttentionInProjection, \
    ScaledDotProduct, MultiheadAttentionOutProjection
from torch.nn.functional import multi_head_attention_forward as mha_forward
from torch.testing import assert_allclose
from ..common.torchtext_test_case import TorchtextTestCase


class TestUtils(TorchtextTestCase):

    def test_multiheadattention(self):
        embed_dim, nhead, tgt_len, src_len, bsz = 10, 5, 6, 10, 64
        # Build torchtext MultiheadAttention models
        q_in = MultiheadAttentionInProjection(embed_dim, nhead)
        k_in = MultiheadAttentionInProjection(embed_dim, nhead)
        v_in = MultiheadAttentionInProjection(embed_dim, nhead)
        MHA_out = MultiheadAttentionOutProjection(embed_dim // nhead, nhead)
        SDP = ScaledDotProduct(nhead)

        query = torch.randn(tgt_len, bsz, embed_dim)
        key = value = torch.randn(src_len, bsz, embed_dim)

        # MultiheadAttention with building blocks
        q = q_in(query)
        k = k_in(key)
        v = v_in(value)
        attn_output, attn_weights = SDP(q, k, v)
        mha_output = MHA_out(attn_output)

        # Use torch.nn.functional.multi_head_attention_forward
        in_proj_weight = torch.cat([q_in.linear.weight, k_in.linear.weight, v_in.linear.weight])
        torch_mha_output, torch_mha_weights = mha_forward(query, key, value,
                                                          embed_dim, nhead,
                                                          in_proj_weight, None,
                                                          None, None, False, 0.0,
                                                          MHA_out.linear.weight, MHA_out.linear.bias)
        assert_allclose(mha_output, torch_mha_output)
        attn_weights = attn_weights.view(bsz, nhead, tgt_len, src_len).sum(dim=1) / nhead
        assert_allclose(attn_weights, torch_mha_weights)
