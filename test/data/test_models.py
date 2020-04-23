import torch
from torchtext.models import MultiheadAttentionContainer, \
    ScaledDotProduct, MultiheadInProject, MultiheadOutProject
from torch.nn.functional import multi_head_attention_forward as mha_forward
from torch.testing import assert_allclose
from ..common.torchtext_test_case import TorchtextTestCase


class TestModels(TorchtextTestCase):

    def test_multiheadattention(self):
        embed_dim, nhead, tgt_len, src_len, bsz = 10, 5, 6, 10, 64
        # Build torchtext MultiheadAttention models
        MHA = MultiheadAttentionContainer((MultiheadInProject(embed_dim, nhead),
                                          MultiheadInProject(embed_dim, nhead),
                                          MultiheadInProject(embed_dim, nhead)),
                                          ScaledDotProduct(),
                                          MultiheadOutProject(embed_dim // nhead, nhead))

        query = torch.rand((tgt_len, bsz, embed_dim))
        key = value = torch.rand((src_len, bsz, embed_dim))
        attn_mask_2D = torch.randint(0, 2, (tgt_len, src_len)).to(torch.bool)
        bias_k = bias_v = torch.rand((1, 1, embed_dim))
        mha_output, attn_weights = MHA(query, key, value,
                                       attn_mask=torch.stack([attn_mask_2D] * (bsz * nhead)),
                                       bias_k=bias_k.repeat(1, bsz, 1).reshape(1, bsz * nhead, -1),
                                       bias_v=bias_v.repeat(1, bsz, 1).reshape(1, bsz * nhead, -1))

        # Use torch.nn.functional.multi_head_attention_forward
        torch_attn_mask = torch.zeros((tgt_len, src_len)).masked_fill_(attn_mask_2D, float('-inf'))
        in_proj_weight = torch.cat([MHA.query_in_proj.proj_layer.weight,
                                    MHA.key_in_proj.proj_layer.weight,
                                    MHA.value_in_proj.proj_layer.weight])
        torch_mha_output, torch_mha_weights = mha_forward(query, key, value,
                                                          embed_dim, nhead,
                                                          in_proj_weight, None,
                                                          bias_k, bias_v,
                                                          False, 0.0,
                                                          MHA.out_proj.proj_layer.weight,
                                                          MHA.out_proj.proj_layer.bias,
                                                          attn_mask=torch_attn_mask)

        assert_allclose(mha_output, torch_mha_output)
        attn_weights = attn_weights.view(bsz, nhead, tgt_len, src_len).sum(dim=1) / nhead
        assert_allclose(attn_weights, torch_mha_weights)

    def test_broadcast_scaled_dot_product(self):
        embed_dim, nhead, tgt_len, src_len, bsz = 10, 5, 6, 10, 64
        SDP = ScaledDotProduct()
        query = torch.rand((tgt_len, 1, embed_dim))
        key = value = torch.rand((src_len, 1, embed_dim))
        attn_mask_2D = torch.randint(0, 2, (tgt_len, src_len)).to(torch.bool)

        sdp_attn_output_full, sdp_attn_weights_full = SDP(query.expand(tgt_len, bsz * nhead, embed_dim),
                                                          key.expand(src_len, bsz * nhead, embed_dim),
                                                          value.expand(src_len, bsz * nhead, embed_dim),
                                                          attn_mask=attn_mask_2D.expand(bsz * nhead, tgt_len, src_len))

        # query has a batch size of 1 while key/value have a batch size of bsz * nhead
        sdp_attn_output, sdp_attn_weights = SDP(query, key.expand(src_len, bsz * nhead, embed_dim),
                                                value.expand(src_len, bsz * nhead, embed_dim),
                                                attn_mask=attn_mask_2D.expand(bsz * nhead, tgt_len, src_len))
        assert_allclose(sdp_attn_output, sdp_attn_output_full)
        assert_allclose(sdp_attn_weights, sdp_attn_weights_full)

        # key/value have a batch size of 1 while query has a batch size of bsz * nhead
        sdp_attn_output, sdp_attn_weights = SDP(query.expand(tgt_len, bsz * nhead, embed_dim),
                                                key, value,
                                                attn_mask=attn_mask_2D.expand(bsz * nhead, tgt_len, src_len))
        assert_allclose(sdp_attn_output, sdp_attn_output_full)
        assert_allclose(sdp_attn_weights, sdp_attn_weights_full)

        # key/value have a size of (3, 3, src_len, bsz * nhead, embed_dim)
        # while query has a size of (tgt_len, 1, embed_dim)
        sdp_attn_output, sdp_attn_weights = SDP(query.expand(tgt_len, 1, embed_dim),
                                                key.expand(3, 3, src_len, bsz * nhead, embed_dim),
                                                value.expand(3, 3, src_len, bsz * nhead, embed_dim),
                                                attn_mask=attn_mask_2D.expand(bsz * nhead, tgt_len, src_len))
        assert list(sdp_attn_output.size()) == [3, 3, tgt_len, bsz * nhead, embed_dim]
        assert list(sdp_attn_weights.size()) == [3, 3, bsz * nhead, tgt_len, embed_dim]
        assert_allclose(sdp_attn_output[2][2], sdp_attn_output_full)
        assert_allclose(sdp_attn_weights[2][2], sdp_attn_weights_full)

        # key/value have a size of (src_len, 1, embed_dim)
        # while query has a size of (1, 2, 3, tgt_len, bsz * nhead, embed_dim)
        sdp_attn_output, sdp_attn_weights = SDP(query.expand(1, 2, 3, tgt_len, bsz * nhead, embed_dim),
                                                key.expand(src_len, 1, embed_dim),
                                                value.expand(src_len, 1, embed_dim),
                                                attn_mask=attn_mask_2D.expand(bsz * nhead, tgt_len, src_len))
        assert list(sdp_attn_output.size()) == [1, 2, 3, tgt_len, bsz * nhead, embed_dim]
        assert list(sdp_attn_weights.size()) == [1, 2, 3, bsz * nhead, tgt_len, embed_dim]
        assert_allclose(sdp_attn_output[0][1][2], sdp_attn_output_full)
        assert_allclose(sdp_attn_weights[0][1][2], sdp_attn_weights_full)

        # attn_mask in a size of (1, tgt_len, src_len)
        # 2D tensor is not supported for attn_mask
        sdp_attn_output, sdp_attn_weights = SDP(query.expand(tgt_len, bsz * nhead, embed_dim),
                                                key.expand(src_len, bsz * nhead, embed_dim),
                                                value.expand(src_len, bsz * nhead, embed_dim),
                                                attn_mask=attn_mask_2D.expand(1, tgt_len, src_len))
        assert_allclose(sdp_attn_output, sdp_attn_output_full)
        assert_allclose(sdp_attn_weights, sdp_attn_weights_full)
