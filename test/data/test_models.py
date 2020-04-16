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
                                          ScaledDotProduct(nhead),
                                          MultiheadOutProject(embed_dim // nhead, nhead))

        query = torch.rand((tgt_len, bsz, embed_dim))
        key = value = torch.rand((src_len, bsz, embed_dim))
        attn_mask_2D = torch.randint(0, 2, (tgt_len, src_len)).to(torch.bool)
        mha_output, attn_weights = MHA(query, key, value,
                                       attn_mask=torch.stack([attn_mask_2D] * (bsz * nhead)))

        # Use torch.nn.functional.multi_head_attention_forward
        torch_attn_mask = torch.zeros((tgt_len, src_len)).masked_fill_(attn_mask_2D, float('-inf'))
        in_proj_weight = torch.cat([MHA.query_in_proj.proj_layer.weight,
                                    MHA.key_in_proj.proj_layer.weight,
                                    MHA.value_in_proj.proj_layer.weight])
        torch_mha_output, torch_mha_weights = mha_forward(query, key, value,
                                                          embed_dim, nhead,
                                                          in_proj_weight, None,
                                                          None, None, False, 0.0,
                                                          MHA.out_proj.proj_layer.weight,
                                                          MHA.out_proj.proj_layer.bias,
                                                          attn_mask=torch_attn_mask)

        assert_allclose(mha_output, torch_mha_output)
        attn_weights = attn_weights.view(bsz, nhead, tgt_len, src_len).sum(dim=1) / nhead
        assert_allclose(attn_weights, torch_mha_weights)
