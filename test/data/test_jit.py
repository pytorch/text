import torch
from torchtext.modules import MultiheadAttentionContainer, \
    ScaledDotProduct, MultiheadInProject, MultiheadOutProject
from torch.testing import assert_allclose
from ..common.torchtext_test_case import TorchtextTestCase


class TestJIT(TorchtextTestCase):

    def test_torchscript_multiheadattention(self):
        embed_dim, nhead, tgt_len, src_len, bsz = 10, 5, 6, 10, 64
        # Build torchtext MultiheadAttention models
        MHA = MultiheadAttentionContainer((MultiheadInProject(embed_dim, nhead),
                                          MultiheadInProject(embed_dim, nhead),
                                          MultiheadInProject(embed_dim, nhead)),
                                          ScaledDotProduct(),
                                          MultiheadOutProject(embed_dim // nhead, nhead))
        query = torch.rand((tgt_len, bsz, embed_dim))
        key = value = torch.rand((src_len, bsz, embed_dim))
        attn_mask = torch.randint(0, 2, (tgt_len, src_len)).to(torch.bool)
        attn_mask = torch.stack([attn_mask] * (bsz * nhead))
        mha_output, attn_weights = MHA(query, key, value, attn_mask=attn_mask)

        ts_MHA = torch.jit.script(MHA)
        ts_mha_output, ts_attn_weights = ts_MHA(query, key, value, attn_mask=attn_mask)
        assert_allclose(mha_output, ts_mha_output)
        assert_allclose(attn_weights, ts_attn_weights)
