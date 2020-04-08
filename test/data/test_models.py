import torch
from torchtext.models import MultiheadAttentionContainer
from torch.nn.functional import multi_head_attention_forward as mha_forward
from torch.testing import assert_allclose
from ..common.torchtext_test_case import TorchtextTestCase


class TestUtils(TorchtextTestCase):

    def test_multiheadattention(self):
        embed_dim, nhead, tgt_len, src_len, bsz = 10, 5, 6, 10, 64
        # Build torchtext MultiheadAttention models
        MHA = MultiheadAttentionContainer(embed_dim, nhead)

        query = torch.rand((tgt_len, bsz, embed_dim))
        key = value = torch.rand((src_len, bsz, embed_dim))
        mha_output, attn_weights = MHA(query, key, value)

        # Use torch.nn.functional.multi_head_attention_forward
        in_proj_weight = torch.cat([MHA.query_in_proj.weight, MHA.key_in_proj.weight, MHA.value_in_proj.weight])
        torch_mha_output, torch_mha_weights = mha_forward(query, key, value,
                                                          embed_dim, nhead,
                                                          in_proj_weight, None,
                                                          None, None, False, 0.0,
                                                          MHA.out_proj.weight, MHA.out_proj.bias)

        assert_allclose(mha_output, torch_mha_output)
        attn_weights = attn_weights.view(bsz, nhead, tgt_len, src_len).sum(dim=1) / nhead
        assert_allclose(attn_weights, torch_mha_weights)
