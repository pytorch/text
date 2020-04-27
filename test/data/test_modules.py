import torch
from torchtext.modules import MultiheadAttentionContainer, ScaledDotProduct
from torch.nn.functional import multi_head_attention_forward as mha_forward
from torch.testing import assert_allclose
from ..common.torchtext_test_case import TorchtextTestCase


class TestModels(TorchtextTestCase):

    def test_multiheadattention(self):
        embed_dim, nhead, tgt_len, src_len, bsz = 10, 5, 6, 10, 64
        # Build torchtext MultiheadAttention module
        MHA = MultiheadAttentionContainer(nhead,
                                          (torch.nn.Linear(embed_dim, embed_dim),
                                           torch.nn.Linear(embed_dim, embed_dim),
                                           torch.nn.Linear(embed_dim, embed_dim),),
                                          ScaledDotProduct(),
                                          torch.nn.Linear(embed_dim, embed_dim))

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
        in_proj_weight = torch.cat([MHA.query_in_proj.weight,
                                    MHA.key_in_proj.weight,
                                    MHA.value_in_proj.weight])
        torch_mha_output, torch_mha_weights = mha_forward(query, key, value,
                                                          embed_dim, nhead,
                                                          in_proj_weight, None,
                                                          bias_k, bias_v,
                                                          False, 0.0,
                                                          MHA.out_proj.weight,
                                                          MHA.out_proj.bias,
                                                          attn_mask=torch_attn_mask)

        assert_allclose(mha_output, torch_mha_output)
        # With bias_k and bias_v, src_len needs to plus 1
        attn_weights = attn_weights.view(bsz, nhead, tgt_len, src_len + 1).sum(dim=1) / nhead
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
        # dim -2 is not equal to neither key/value's dim -2 or 1
        with self.assertRaises(RuntimeError):
            SDP(query.expand(tgt_len, 1, embed_dim), key.expand(3, 3, src_len, bsz * nhead, embed_dim),
                value.expand(3, 3, src_len, bsz * nhead, embed_dim),
                attn_mask=attn_mask_2D.expand(bsz * nhead, tgt_len, src_len)

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
        # key dim -2 is not equal to value dim -2
        with self.assertRaisesRegex(AssertionError, "Shape of key, value must match"):
            SDP(query.expand(1, 2, 3, tgt_len, bsz * nhead, embed_dim), key.expand(src_len, 2, embed_dim),
                value.expand(src_len, 1, embed_dim),
                attn_mask=attn_mask_2D.expand(bsz * nhead, tgt_len, src_len))
        # key/value dim -2 is not equal to neither query's dim -2 or 1
        with self.assertRaises(RuntimeError):
            SDP(query.expand(1, 2, 3, tgt_len, bsz * nhead, embed_dim), key.expand(src_len, 2, embed_dim),
                value.expand(src_len, 2, embed_dim),
                attn_mask=attn_mask_2D.expand(bsz * nhead, tgt_len, src_len))

        # attn_mask in a size of (1, tgt_len, src_len)
        # 2D tensor is not supported for attn_mask
        sdp_attn_output, sdp_attn_weights = SDP(query.expand(tgt_len, bsz * nhead, embed_dim),
                                                key.expand(src_len, bsz * nhead, embed_dim),
                                                value.expand(src_len, bsz * nhead, embed_dim),
                                                attn_mask=attn_mask_2D.expand(1, tgt_len, src_len))
        assert_allclose(sdp_attn_output, sdp_attn_output_full)
        assert_allclose(sdp_attn_weights, sdp_attn_weights_full)
        # attn_mask's dim -3 is not equal to neither batch size or 1
        with self.assertRaisesRegex(RuntimeError, "The size of the attn_mask is not correct."):
            SDP(query.expand(tgt_len, bsz * nhead, embed_dim), key.expand(src_len, bsz * nhead, embed_dim),
                value.expand(src_len, bsz * nhead, embed_dim),
                attn_mask=attn_mask_2D.expand(2, tgt_len, src_len))
